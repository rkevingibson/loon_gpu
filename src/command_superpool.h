#pragma once

#include "containers.h"
#include "platform_utils.h"
namespace loon::gpu {

/**
 * @brief CommandSuperpool - a helper for managing command pools.
 *
 * Ideally we would like to reuse command pools across command buffers, and reset command pools all
 * at once rather than resetting individual command buffers. The CommandSuperpool enables that.
 *
 * We have a conservative upper-bound on the maximum number of command pools we can create, as
 * memory is pre-allocated. This should be fine for all reasonable applications - we need one
 * command pool for each command buffer being recording concurrently, per frame in flight.
 *
 * For workloads that don't follow a traditional frame loop (i.e. pure compute workloads), we place
 * a limit on the number of command buffers that can be recorded per pool. When this limit is
 * reached, the command pool can be reused when the last command in it is retired (as determined by
 * the timeline indexes passed to `on_queue_submit()`)
 *
 * Internally, command pools are kept in a Vector that's been reserved upfront - it should never
 * resize/reallocate to ensure pointer stability. Each element of this vector can also be in one of
 * 3 linked lists depending on their current state: One for currently "locked" pools (unavailable
 * for recording), one for available pools, and one for available pools that were already used in
 * the current frame. These lists are treated as lock-free stacks.
 *
 * Unless otherwise noted, all operations are thread-safe, and intended to be lock-free in the happy
 * path.
 * @tparam PoolType
 */

template <class PoolType>
class CommandSuperpool {
   public:
    CommandSuperpool() = default;
    CommandSuperpool(Allocator alloc, uint32_t max_command_pools) :
        m_nodes(alloc, max_command_pools) {}

    static constexpr uint32_t kMaxCommandBuffersPerPool = 32;

    /**
     * @brief Get a command pool that can be used to record a command buffer, or nullptr if all
     * command pools are already being used.
     *
     * @return PoolType*
     */
    PoolType* acquire_command_pool();

    /**
     * @brief Return a command pool to the superpool, so that it can be reused.
     *
     * @param pool
     */
    void release_command_pool(PoolType* pool);

    /**
     * @brief Signal that a frame is being presented. Command pools used this frame will be locked
     * until they are safe to reuse.
     *
     */
    void end_of_frame();

    /**
     * @brief Signal that a command buffer has been submitted to the queue. This lets the superpool
     * check if any command pools are ready to be reused.
     *
     * @param submitted_timeline_idx
     * @param completed_timeline_idx
     */
    void on_queue_submit(uint64_t submitted_timeline_idx, uint64_t completed_timeline_idx);

    // Not thread-safe: Used for cleanup on shutdown.
    void visit_pools(Function<void, const PoolType&> fn) const {
        for (auto& n : m_nodes) { fn(n.pool); }
    }

   private:
    struct alignas(alignof(int64_t)) NodeHead {
        uint32_t aba   = 0;
        uint32_t index = kInvalidIdx;
        int64_t  as_int() const noexcept {
            int64_t result;
            memcpy(&result, this, sizeof(int64_t));
            return result;
        }

        static NodeHead atomic_load(NodeHead* h) {
            const int64_t x = loon::gpu::atomic_load(reinterpret_cast<int64_t*>(h));
            NodeHead      res;
            memcpy(&res, &x, sizeof(int64_t));
            return res;
        }

        static NodeHead atomic_exchange(NodeHead* h, const NodeHead& val) {
            const int64_t x =
                loon::gpu::atomic_exchange(reinterpret_cast<int64_t*>(h), val.as_int());
            NodeHead res;
            memcpy(&res, &x, sizeof(int64_t));
            return res;
        }
    };

    struct Node {
        PoolType pool{};
        uint32_t next_idx              = 0;
        uint32_t times_used_this_frame = 0;
        uint64_t timeline_idx = 0;  // Timeline index that was last seen before this node was locked
                                    // - i.e. when it is safe to reuse.
    };

    Node* try_pop_stack(NodeHead* head) {
        NodeHead next;
        NodeHead orig = NodeHead::atomic_load(head);
        do {
            if (orig.index == kInvalidIdx) { return nullptr; }
            next = {
                .aba   = orig.aba + 1,
                .index = m_nodes[orig.index].next_idx,
            };

        } while (!atomic_compare_exchange(reinterpret_cast<int64_t*>(head),
                                          reinterpret_cast<int64_t*>(&orig),
                                          next.as_int()));
        return &m_nodes[orig.index];
    }

    void push_stack(NodeHead* head, Node* node) {
        NodeHead next;
        NodeHead orig     = NodeHead::atomic_load(head);
        uint32_t node_idx = node - m_nodes.data();
        do {
            node->next_idx = orig.index;
            next           = {
                          .aba   = orig.aba + 1,
                          .index = node_idx,
            };
        } while (!atomic_compare_exchange(reinterpret_cast<int64_t*>(head),
                                          reinterpret_cast<int64_t*>(&orig),
                                          next.as_int()));
    }

    Node* allocate_node() {
        mutex_lock(&m_mutex);
        if (m_allocated_nodes_count == m_nodes.capacity()) { return nullptr; }
        m_nodes.emplace_back();  // Since we reserved upfront in the constructor, this won't move
                                 // any pointers.
        Node* result = &m_nodes[m_allocated_nodes_count++];
        mutex_unlock(&m_mutex);
        return result;
    }

    static constexpr uint32_t kInvalidIdx = ~0u;

    NodeHead m_current_frame_nodes_head;
    NodeHead m_available_nodes_head;
    NodeHead m_locked_nodes_head;
    uint32_t m_allocated_nodes_count = 0;
    uint64_t m_current_timeline_idx  = 0;
    mutex    m_mutex = LOON_MUTEX_INIT;  // We should only need to lock when creating a new command
                                         // pool. In steady state usage, we should be lock free.
    Vector<Node> m_nodes;                // We never grow this, but instead just pre-allocate to a
};

template <class P>
P* CommandSuperpool<P>::acquire_command_pool() {
    Node* n = try_pop_stack(&m_current_frame_nodes_head);
    if (n == nullptr) {
        n = try_pop_stack(&m_available_nodes_head);
        if (n) {
            n->pool.reset();
            n->times_used_this_frame = 0;
        }
    }
    if (n == nullptr) {
        n = allocate_node();
        if (n) n->times_used_this_frame = 0;
    }
    if (n == nullptr) { return nullptr; }
    n->times_used_this_frame++;
    return &n->pool;
}

template <class P>
void CommandSuperpool<P>::release_command_pool(P* p) {
    Node* node = reinterpret_cast<Node*>(p);
    if (node->times_used_this_frame == kMaxCommandBuffersPerPool) {
        push_stack(&m_locked_nodes_head, node);
    } else {
        push_stack(&m_current_frame_nodes_head, reinterpret_cast<Node*>(p));
    }
}

template <class P>
void CommandSuperpool<P>::end_of_frame() {
    const uint64_t timeline_idx = atomic_load(reinterpret_cast<int64_t*>(&m_current_timeline_idx));

    // We grab the entire list, then iterate it locally, moving nodes to m_locked_nodes and setting
    // the right timeline_idx for them.
    NodeHead list = NodeHead::atomic_exchange(&m_current_frame_nodes_head,
                                              NodeHead{.aba = 0, .index = kInvalidIdx});

    for (uint32_t idx = list.index; idx != kInvalidIdx;) {
        const uint32_t next_idx   = m_nodes[idx].next_idx;
        m_nodes[idx].timeline_idx = timeline_idx;
        push_stack(&m_locked_nodes_head, &m_nodes[idx]);
        idx = next_idx;
    }
}

template <class P>
void CommandSuperpool<P>::on_queue_submit(uint64_t submitted_timeline_idx,
                                          uint64_t completed_timeline_idx) {
    atomic_exchange(reinterpret_cast<int64_t*>(&m_current_timeline_idx),
                    static_cast<int64_t>(submitted_timeline_idx));
    (void)(completed_timeline_idx);

    // For simplicity, we grab a copy of the locked_nodes list and insert the nodes based on if
    // they're completed or not.
    NodeHead list =
        NodeHead::atomic_exchange(&m_locked_nodes_head, NodeHead{.aba = 0, .index = kInvalidIdx});

    for (uint32_t idx = list.index; idx != kInvalidIdx;) {
        const uint32_t next_idx = m_nodes[idx].next_idx;
        Node&          n        = m_nodes[idx];
        if (n.timeline_idx <= completed_timeline_idx) {
            push_stack(&m_available_nodes_head, &m_nodes[idx]);
        } else {
            push_stack(&m_locked_nodes_head, &m_nodes[idx]);
        }
        idx = next_idx;
    }
}

}  // namespace loon::gpu