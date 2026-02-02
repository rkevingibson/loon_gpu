import lldb


class ArraySynthBase(object):
    """
    Helper baseclass aimed to reduce the boilerplate needed for "array-like" containers
    """

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj

    def bind_to(self, pointer, size):
        array_p = pointer.GetType().GetPointeeType().GetArrayType(size).GetPointerType()
        self.array = pointer.Cast(array_p).Dereference()

    def update(self):
        self.array = self.valobj

    def num_children(self, max_children):
        return self.array.GetNumChildren(max_children)

    def get_child_index(self, name):
        return self.array.GetIndexOfChildWithName(name)

    def get_child_at_index(self, index):
        return self.array.GetChildAtIndex(index)

    def has_children(self):
        return self.array.MightHaveChildren()

    def get_value(self):
        return self.array


class SpanSynth(ArraySynthBase):
    def update(self):
        self.size = self.valobj.GetChildMemberWithName("m_len").GetValueAsUnsigned()
        data = self.valobj.GetChildMemberWithName("m_ptr")
        self.element_type = data.GetType().GetPointeeType().GetName()
        self.bind_to(data, int(self.size))

    def get_summary(self):
        return f"Span<{self.element_type}> (size={self.size})"


class StringViewSynth(ArraySynthBase):
    def update(self):
        data = self.valobj.GetChildMemberWithName("m_ptr")
        self.size = self.valobj.GetChildMemberWithName("m_len").GetValueAsUnsigned()
        self.bind_to(data, int(self.size))

    def get_summary(self):
        return f"{self.array}"


def __lldb_init_module(debugger, internal_dict):
    """
    This function will be automatically called by LLDB when the module is loaded, here
    we register the various synthetics/summaries we have build before
    """

    category_name = "loon_gpu"
    category = debugger.GetCategory(category_name)

    # Make sure we don't accidentally keep accumulating languages or override the user's
    # category enablement in Xcode, where lldb-rpc-server loads this file once for eac
    # debugging session
    if not category.IsValid():
        category = debugger.CreateCategory(category_name)
        category.AddLanguage(lldb.eLanguageTypeC_plus_plus)
        category.SetEnabled(True)

    def add_summary(typename, impl):
        summary = None

        if isinstance(impl, str):
            summary = lldb.SBTypeSummary.CreateWithSummaryString(impl)
            summary.SetOptions(lldb.eTypeOptionCascade)
        else:
            # Unfortunately programmatic summary string generation is an entirely different codepath
            # in LLDB. Register a convenient trampoline function which makes it look like it's part
            # of the SyntheticChildrenProvider contract
            summary = lldb.SBTypeSummary.CreateWithScriptCode(f"""
				synth = {impl.__module__}.{impl.__qualname__}(valobj.GetNonSyntheticValue(), internal_dict)
				synth.update()

				return synth.get_summary()
			""")
            summary.SetOptions(
                lldb.eTypeOptionCascade | lldb.eTypeOptionFrontEndWantsDereference
            )

        category.AddTypeSummary(lldb.SBTypeNameSpecifier(typename, True), summary)

    def add_synthetic(typename, impl):
        add_summary(typename, impl)

        synthetic = lldb.SBTypeSynthetic.CreateWithClassName(
            f"{impl.__module__}.{impl.__qualname__}"
        )
        synthetic.SetOptions(
            lldb.eTypeOptionCascade | lldb.eTypeOptionFrontEndWantsDereference
        )

        category.AddTypeSynthetic(lldb.SBTypeNameSpecifier(typename, True), synthetic)

    add_synthetic("^loon::gpu::Span<.+>$", SpanSynth)
    add_synthetic("^loon::StringView$", StringViewSynth)

    add_summary("^geometry::float2$", "x=${var.x} y=${var.y}")
    add_summary("^geometry::float3$", "x=${var.x} y=${var.y} z=${var.z}")
    add_summary("^geometry::float4$", "x=${var.x} y=${var.y} z=${var.z} w=${var.w}")
