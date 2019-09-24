use nannou::prelude::*;
use std::thread::sleep;
use std::time;

fn main() {
    nannou::app(model)
        .update(update)
        .run()
}

struct Model {
    training_data_in: [[f32; 4]; 16],
    training_data_out: [[f32; 9]; 16],
    time: usize,
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

    let training_data_in: [[f32; 4]; 16] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ] ;

    let training_data_out: [[f32; 9]; 16] = [
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ] ;

    let time = 15;

    Model {
        training_data_in,
        training_data_out,
        time,
        _window }
}

fn update(_app: &App, model: &mut Model, _update: Update) {

    model.time += 1;

    if model.time > 15 {
        model.time = 0;
    }

    sleep(time::Duration::new(1, 0));

}

fn view(app: &App, model: &Model, frame: &Frame) {

    let draw = app.draw();

    let t = model.time;

    let tdi = model.training_data_in[t];
    let tdu = model.training_data_out[t];

    draw.ellipse().x_y(-100.0, 5.0).radius(10.0).color(rgb(tdi[0], tdi[0], tdi[0]));
    draw.ellipse().x_y(-100.0, 62.0).radius(10.0).color(rgb(tdi[1], tdi[1], tdi[1]));
    draw.ellipse().x_y(-100.0, 123.0).radius(10.0).color(rgb(tdi[2], tdi[2], tdi[2]));
    draw.ellipse().x_y(-100.0, 192.0).radius(10.0).color(rgb(tdi[3], tdi[3], tdi[3]));

    draw.rect().x_y(50.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[0], tdu[0], tdu[0]));
    draw.rect().x_y(50.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[1], tdu[1], tdu[1]));

    draw.rect().x_y(135.0, 192.0).w_h(44.0, 16.0).color(rgb(tdu[2], tdu[2], tdu[2]));
    draw.rect().x_y(165.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[3], tdu[3], tdu[3]));
    draw.rect().x_y(165.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[4], tdu[4], tdu[4]));
    draw.rect().x_y(135.0, 8.0).w_h(44.0, 16.0).color(rgb(tdu[5], tdu[5], tdu[5]));
    draw.rect().x_y(100.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[6], tdu[6], tdu[6]));
    draw.rect().x_y(100.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[7], tdu[7], tdu[7]));
    draw.rect().x_y(135.0, 100.0).w_h(44.0, 16.0).color(rgb(tdu[8], tdu[8], tdu[8]));

    draw.to_frame(app, frame).unwrap();
}
