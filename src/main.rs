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
    let draw = app.draw();

    draw.rect().x_y(100.0, 100.0).w_h(100.0, 15.0).color(WHITE);
    draw.rect().x_y(100.0, 115.0).w_h(15.0, 100.0).color(WHITE);
    draw.rect().x_y(200.0, 115.0).w_h(15.0, 100.0).color(WHITE);
    draw.rect().x_y(100.0, 120.0).w_h(100.0, 15.0).color(WHITE);
    draw.rect().x_y(100.0, 135.0).w_h(15.0, 100.0).color(WHITE);
    draw.rect().x_y(200.0, 135.0).w_h(15.0, 100.0).color(WHITE);
    draw.rect().x_y(100.0, 140.0).w_h(100.0, 15.0).color(WHITE);

    draw.to_frame(app, frame).unwrap();
}
