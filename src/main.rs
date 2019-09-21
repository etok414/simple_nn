use nannou::prelude::*;
use std::thread::sleep;
use std::time;

fn main() {
    nannou::app(model).run()
}

struct Model {
    training_data_in: [[f32; 4]; 16],
    training_data_out: [[f32; 9]; 16],
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
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],


    ] ;

    Model {
        training_data_in,
        training_data_out,
        _window }
}

fn view(app: &App, model: &Model, frame: &Frame) {
    let draw = app.draw();

    for n in 0..16 {
        let tdi = model.training_data_in[n];
        let tdu = model.training_data_out[n];

        draw.ellipse().x_y(0.0, 50.0).radius(10.0).color(rgb(tdi[0], tdi[0], tdi[0]));
        draw.ellipse().x_y(0.0, 100.0).radius(10.0).color(rgb(tdi[1], tdi[1], tdi[1]));
        draw.ellipse().x_y(0.0, 150.0).radius(10.0).color(rgb(tdi[2], tdi[2], tdi[2]));
        draw.ellipse().x_y(0.0, 200.0).radius(10.0).color(rgb(tdi[3], tdi[3], tdi[3]));

        draw.rect().x_y(120.0, 180.0).w_h(50.0, 15.0).color(rgb(tdu[2], tdu[2], tdu[2]));
        draw.rect().x_y(90.0, 145.0).w_h(15.0, 50.0).color(rgb(tdu[7], tdu[7], tdu[7]));
        draw.rect().x_y(150.0, 145.0).w_h(15.0, 50.0).color(rgb(tdu[3], tdu[3], tdu[3]));
        draw.rect().x_y(120.0, 110.0).w_h(50.0, 15.0).color(rgb(tdu[8], tdu[8], tdu[8]));
        draw.rect().x_y(150.0, 75.0).w_h(15.0, 50.0).color(rgb(tdu[4], tdu[4], tdu[4]));
        draw.rect().x_y(90.0, 75.0).w_h(15.0, 50.0).color(rgb(tdu[6], tdu[6], tdu[6]));
        draw.rect().x_y(120.0, 40.0).w_h(50.0, 15.0).color(rgb(tdu[5], tdu[5], tdu[5]));

        draw.rect().x_y(40.0, 75.0).w_h(15.0, 70.0).color(rgb(tdi[0], tdi[0], tdi[0]));
        draw.rect().x_y(40.0, 145.0).w_h(15.0, 70.0).color(rgb(tdi[1], tdi[1], tdi[1]));

        draw.to_frame(app, frame).unwrap();

        sleep(time::Duration::new(1, 0));
    }
}
