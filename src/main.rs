use nannou::prelude::*;

fn main() {
    nannou::app(model).run()
}

struct Model {
    _window: WindowId,
}

fn model(app: &App) -> Model {
    let _window = app
    .new_window()
    .with_dimensions(700, 700)
    .with_title("Simple Neural Network")
    .view(view)
    .build()
    .unwrap();
    Model { _window }
}

fn view(app: &App, _model: &Model, frame: &Frame) {
    draw.rect().x_y(100, 100).w_h(100, 10)


    draw.to_frame(app, frame).unwrap();
}
