
#include "example.h"
#include <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#import <Cocoa/Cocoa.h>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "common/example.h"
#import "common/shaders.h"
#import <memory>
#import <string>

// MAKE: AppViewController

@interface AppViewController : NSViewController <NSWindowDelegate> {
  WindowState window_state;
  std::unique_ptr<Example> current_example;
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
    std::string shader_search_path = "/Users/kevin/Code/loon_gpu/examples/";

    self->window_state = {
        .native_window_handle = 0,
        .native_instance_handle = 0,
        .width = 1200,
        .height = 800,
        .shader_loader =
            std::make_unique<ShaderLoader>(shader_search_path.c_str()),
    };

    self->selected_example = ExampleName::TexturedCube;
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
  [NSApp activateIgnoringOtherApps:YES];
}

- (void)drawInMTKView:(nonnull MTKView *)view {

  CAMetalLayer *layer = (CAMetalLayer *)view.layer;
  self->window_state.native_window_handle = reinterpret_cast<uintptr_t>(layer);
  auto size = layer.drawableSize;
  window_state.width = static_cast<uint16_t>(size.width);
  window_state.height = static_cast<uint16_t>(size.height);
  if (!current_example) {
    current_example = create_example(selected_example, self->window_state);
  }
  current_example->Update(self->window_state);
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
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