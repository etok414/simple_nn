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

    draw.rect().x_y(120.0, 180.0).w_h(50.0, 15.0).color(WHITE);
    draw.rect().x_y(90.0, 145.0).w_h(15.0, 50.0).color(WHITE);
    draw.rect().x_y(150.0, 145.0).w_h(15.0, 50.0).color(WHITE);
    draw.rect().x_y(120.0, 110.0).w_h(50.0, 15.0).color(WHITE);
    draw.rect().x_y(150.0, 75.0).w_h(15.0, 50.0).color(WHITE);
    draw.rect().x_y(90.0, 75.0).w_h(15.0, 50.0).color(WHITE);
    draw.rect().x_y(120.0, 40.0).w_h(50.0, 15.0).color(WHITE);

    draw.rect().x_y(40.0, 75.0).w_h(15.0, 50.0).color(WHITE);
    draw.rect().x_y(40.0, 145.0).w_h(15.0, 50.0).color(WHITE);

    draw.to_frame(app, frame).unwrap();
}
