// This is a workaround because by default Slang assigns bindings assuming you can use VK_EXT_mutable_descriptor_type, which we don't want to require.
export T getDescriptorFromHandle<T>(DescriptorHandle<T> handle) where T : IOpaqueDescriptor
{
    return defaultGetDescriptorFromHandle(handle, BindlessDescriptorOptions.None);
}