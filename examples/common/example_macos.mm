
#import "example.h"
#import "gpu/loon_gpu.h"
#import "imgui/imgui.h"
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "common/box.h"
#import "common/example.h"
#import "common/shaders.h"
#import "imgui/imgui_impl_osx.h"
#import <string>

// MAKE: AppViewController

@interface AppViewController : NSViewController <NSWindowDelegate> {
  WindowState window_state;
  loon::Box<Example> current_example;
  ExampleName selected_example;
}
@end

@interface AppViewController () <MTKViewDelegate>
@property(nonatomic, readonly) MTKView *mtkView;
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@end

@implementation AppViewController

- (instancetype)initWithNibName:(NSNibName)nibNameOrNil
                         bundle:(NSBundle *)nibBundleOrNil {
  self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
  _device = MTLCreateSystemDefaultDevice();
  if (self) {
    auto file_paths = default_file_paths();
    self->window_state = {
        .native_window_handle = 0,
        .native_instance_handle = 0,
        .width = 1200,
        .height = 800,
        .shader_loader = loon::make_box<ShaderLoader>(
            file_paths.shader_directory.c_str(),
            loon::gpu::device_backend() == loon::gpu::Backend::Metal),
        .file_paths = file_paths,
    };

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();

    self->selected_example = ExampleName::HelloCube;
  }

  return self;
}

- (MTKView *)mtkView {
  return (MTKView *)self.view;
}

- (void)loadView {
  self.view = [[MTKView alloc] initWithFrame:CGRectMake(0, 0, 1200, 800)];
}

- (void)viewDidLoad {
  [super viewDidLoad];
  self.mtkView.device = self.device;
  self.mtkView.delegate = self;
  ImGui_ImplOSX_Init(self.view);
  [NSApp activateIgnoringOtherApps:YES];
}

- (void)drawInMTKView:(nonnull MTKView *)view {

  CAMetalLayer *layer = (CAMetalLayer *)view.layer;
  self->window_state.native_window_handle = reinterpret_cast<uintptr_t>(layer);
  auto size = layer.drawableSize;
  window_state.width = static_cast<uint16_t>(size.width);
  window_state.height = static_cast<uint16_t>(size.height);

  ImGuiIO &io = ImGui::GetIO();
  io.DisplaySize.x = view.bounds.size.width;
  io.DisplaySize.y = view.bounds.size.height;
  io.DisplayFramebufferScale = ImVec2(view.window.screen.backingScaleFactor,
                                      view.window.screen.backingScaleFactor);

  if (!current_example) {
    current_example = create_example(selected_example, self->window_state);
  }

  ImGui_ImplOSX_NewFrame(self.view);
  current_example->Update(self->window_state);
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
}

- (bool)acceptsFirstResponder {
  return YES;
}

- (void)keyDown:(NSEvent *)event {
  constexpr uint16_t keyCodeR = 15;
  constexpr uint16_t keyCodeN = 45;
  constexpr uint16_t keyCodeM = 46;
  if (event.isARepeat) {
    return;
  }

  if (event.keyCode == keyCodeR) {
    current_example.reset();
    // Need to reset the shader loader to avoid caching.
    window_state.shader_loader->reset_cache();
    current_example = create_example(selected_example, self->window_state);
  } else if (event.keyCode == keyCodeN) { // Previous example
    current_example.reset();
    window_state.shader_loader->reset_cache();
    selected_example = ExampleName((static_cast<int>(selected_example) +
                                    static_cast<int>(ExampleName::Count) - 1) %
                                   static_cast<int>(ExampleName::Count));
    current_example = create_example(selected_example, window_state);
  } else if (event.keyCode == keyCodeM) { // Next example.
    current_example.reset();
    window_state.shader_loader->reset_cache();
    selected_example = ExampleName((static_cast<int>(selected_example) + 1) %
                                   static_cast<int>(ExampleName::Count));
    current_example = create_example(selected_example, window_state);
  }
}

- (void)keyUp:(NSEvent *)event {
}

@end

// MARK: AppDelegate

@interface AppDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSWindow *window;
@end

@implementation AppDelegate

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  return YES;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    NSViewController *rootViewController =
        [[AppViewController alloc] initWithNibName:nil bundle:nil];
    self.window = [NSWindow windowWithContentViewController:rootViewController];
    [self.window center];
    [self.window makeKeyAndOrderFront:self];
    [self.window makeFirstResponder:rootViewController];
  }
  return self;
}

@end

int main(int argc, char **argv) {
  @autoreleasepool {
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    AppDelegate *appDelegate = [[AppDelegate alloc] init];
    [NSApp setDelegate:appDelegate];

    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];
  }
  return 0;
}
