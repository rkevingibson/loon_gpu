# Spec Non-compliance

For the most part I aim for compliance with the spec published here: https://www.w3.org/TR/webgpu. This is not to say that we are 100% spec compliant - there are bugs and incomplete implementations throughout, that will hopefully go away over time.

However, there are a few differences mostly related to validation that we don't currently intend to support, due to performance or implementation reasons. We list them here.

- When mapping WGPUBuffers, validation suggests `getMappedRange()` can only be called once for a given non-overlapping region of the mapped area, and will fail validation if called multiple times. This would require tracking ranges in an dynamically allocated array, so we choose not to implement it.

