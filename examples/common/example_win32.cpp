
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <shellapi.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/example.h"
#include "common/shaders.h"
#include "imgui/imgui_impl_loon.h"
#include "imgui/imgui_impl_win32.h"



enum ButtonState : uint8_t {
    kButtonStateDefault  = 0x0000,
    kButtonStatePressed  = 0x0001,
    kButtonStateReleased = 0x0002,
    kButtonStateDown     = 0x0004,
};

struct RawInput {
    ButtonState keys[256] = {kButtonStateDefault};  //

    void update() {
        BYTE keyboard_state[256] = {0};
        GetKeyboardState(keyboard_state);

        // Remove pressed/released states
        for (auto& k : keys) {
            k = static_cast<ButtonState>(k & ~(kButtonStatePressed | kButtonStateReleased));
        }

        auto& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) {
            for (int i = 0; i < 256; ++i) { keys[i] = kButtonStateDefault; }
        } else {
            for (int i = 0; i < 256; ++i) {
                if ((keys[i] & kButtonStateDown) && (keyboard_state[i] & 0x80) == 0) {
                    // Was down last frame, no longer down
                    keys[i] = kButtonStateReleased;
                } else if ((keys[i] & kButtonStateDown) == 0 && (keyboard_state[i] & 0x80)) {
                    keys[i] = static_cast<ButtonState>(kButtonStateDown | kButtonStatePressed);
                }
            }
        }
    }
};

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND   hWnd,
                                                             UINT   msg,
                                                             WPARAM wParam,
                                                             LPARAM lParam);

LRESULT WINAPI window_proc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(wnd, msg, wParam, lParam)) { return true; }

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
    ImGui_ImplWin32_EnableDpiAwareness();
    float main_scale = ImGui_ImplWin32_GetDpiScaleForMonitor(
        ::MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));


    WindowState window_state = {
        .file_paths = default_file_paths(),
    };

    // Parse command line args.
    int     argc;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);


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
            window_state.file_paths.shader_directory = std::string(bytes.begin(), bytes.end());
        }
    }

    window_state.shader_loader
        = std::make_unique<ShaderLoader>(window_state.file_paths.shader_directory.c_str());

    const char  CLASS_NAME[] = "LoonWebGPU Examples";
    WNDCLASSEXA wc           = {.cbSize        = sizeof(WNDCLASSEXA),
                                .style         = CS_VREDRAW | CS_HREDRAW,
                                .lpfnWndProc   = window_proc,
                                .cbClsExtra    = 0,
                                .cbWndExtra    = 0,
                                .hInstance     = hInstance,
                                .hIcon         = NULL,
                                .hCursor       = NULL,
                                .hbrBackground = CreateSolidBrush(RGB(30, 30, 30)),
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

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);
    style.FontScaleDpi = main_scale;

    ImGui_ImplWin32_Init(window);

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

        ImGui_ImplWin32_NewFrame();

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

    ImGui_ImplWin32_Shutdown();

    return 0;
}