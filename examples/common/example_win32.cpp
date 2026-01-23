#include <Windows.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/example.h"
#include "common/shaders.h"


enum ButtonState : uint8_t {
    kButtonStateDefault  = 0x0000,
    kButtonStatePressed  = 0x0001,
    kButtonStateReleased = 0x0002,
    kButtonStateDown     = 0x0004,
};

struct RawInput {
    ButtonState keys[256] = {kButtonStateDefault};  //

    void update() {
        BYTE keyboard_state[256];
        GetKeyboardState(keyboard_state);

        // Remove pressed/released states
        for (auto& k : keys) {
            k = static_cast<ButtonState>(k & ~(kButtonStatePressed | kButtonStateReleased));
        }

        for (int i = 0; i < 256; ++i) {
            if ((keys[i] & kButtonStateDown) && (keyboard_state[i] & 0x80) == 0) {
                // Was down last frame, no longer down
                keys[i] = kButtonStateReleased;
            } else if ((keys[i] & kButtonStateDown) == 0 && (keyboard_state[i] & 0x80)) {
                keys[i] = static_cast<ButtonState>(kButtonStateDown | kButtonStatePressed);
            }
        }
    }
};

LRESULT WINAPI window_proc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            CREATESTRUCT* pCreate = (CREATESTRUCT*)(lParam);
            SetWindowLongPtrA(wnd, GWLP_USERDATA, (intptr_t)pCreate->lpCreateParams);
            return 0;
        }
        case WM_PAINT: {
            ValidateRect(wnd, NULL);
            return 0;
        }
        case WM_SIZE: {
            const uint16_t newWidth  = LOWORD(lParam);
            const uint16_t newHeight = HIWORD(lParam);
            WindowState*   win_state = (WindowState*)GetWindowLongPtrA(wnd, GWLP_USERDATA);
            win_state->height        = newHeight;
            win_state->width         = newWidth;
        } break;
        case WM_DESTROY: {
            PostQuitMessage(0);
        } break;
        default: return DefWindowProcA(wnd, msg, wParam, lParam);
    }
    return 0;
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd) {
    WindowState window_state = {};

    // Parse command line args.
    int     argc;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::string shader_search_path = "";

    for (int arg_idx = 0; arg_idx < argc; ++arg_idx) {
        if (std::wstring_view(argv[arg_idx]) == L"--shader_dir" && arg_idx + 1 < argc) {
            int string_size
                = WideCharToMultiByte(CP_UTF8, 0, argv[arg_idx + 1], -1, nullptr, 0, NULL, NULL);

            std::vector<char> bytes(string_size, 0);
            WideCharToMultiByte(CP_UTF8,
                                0,
                                argv[arg_idx + 1],
                                -1,
                                bytes.data(),
                                (int)bytes.size(),
                                NULL,
                                NULL);
            shader_search_path         = std::string(bytes.data());
            window_state.shader_loader = std::make_unique<ShaderLoader>(shader_search_path.c_str());
        }
    }

    // TODO: If ShaderLoader is uninitialized, create a default one which searches for shaders next
    // to the binary location.

    const char  CLASS_NAME[] = "LoonWebGPU Examples";
    WNDCLASSEXA wc           = {.cbSize        = sizeof(WNDCLASSEXA),
                                .style         = CS_VREDRAW | CS_HREDRAW,
                                .lpfnWndProc   = window_proc,
                                .cbClsExtra    = 0,
                                .cbWndExtra    = 0,
                                .hInstance     = hInstance,
                                .hIcon         = NULL,
                                .hCursor       = NULL,
                                .hbrBackground = NULL,
                                .lpszMenuName  = NULL,
                                .lpszClassName = CLASS_NAME,
                                .hIconSm       = NULL};

    RegisterClassExA(&wc);

    HWND window = CreateWindowExA(0,
                                  CLASS_NAME,
                                  "LoonWebGPU Examples",
                                  WS_OVERLAPPEDWINDOW,
                                  CW_USEDEFAULT,
                                  CW_USEDEFAULT,
                                  CW_USEDEFAULT,
                                  CW_USEDEFAULT,
                                  NULL,
                                  NULL,
                                  hInstance,
                                  &window_state);

    window_state.native_window_handle   = (uintptr_t)window;
    window_state.native_instance_handle = (uintptr_t)GetModuleHandle(nullptr);

    ShowWindow(window, SW_NORMAL);
    UpdateWindow(window);

    ExampleName selected_example = ExampleName::TexturedCube;

    std::unique_ptr<Example> current_example = create_example(selected_example, window_state);

    RawInput input;

    bool done = false;
    while (!done) {
        MSG msg = {};
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
            if (msg.message == WM_QUIT) { done = true; }
        }

        if (done) { break; }

        if (IsIconic(window)) {
            Sleep(10);
            continue;
        }

        input.update();

        if (input.keys['R'] & kButtonStatePressed) {
            current_example.reset();
            // Need to reset the shader loader to avoid caching.
            window_state.shader_loader->reset_cache();
            current_example = create_example(selected_example, window_state);
        } else if (input.keys['N'] & kButtonStatePressed) {
            current_example.reset();
            window_state.shader_loader->reset_cache();
            selected_example = ExampleName(
                (static_cast<int>(selected_example) + static_cast<int>(ExampleName::Count) - 1)
                % static_cast<int>(ExampleName::Count));
            current_example = create_example(selected_example, window_state);
        } else if (input.keys['M'] & kButtonStatePressed) {
            current_example.reset();
            window_state.shader_loader->reset_cache();
            selected_example = ExampleName((static_cast<int>(selected_example) + 1)
                                           % static_cast<int>(ExampleName::Count));
            current_example  = create_example(selected_example, window_state);
        }

        current_example->Update(window_state);
    }


    return 0;
}