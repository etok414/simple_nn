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

struct Node {
    bias: f32,
    weights: Vec<f32>,
}

impl Node {
    pub fn new(number_of_weights: usize) -> Node {
        // Should probably be random.
        Node {bias: 0.0, weights: vec![0.0; number_of_weights]}
    }

    pub fn calculate(&self, previous_layer:&Vec<f32>) -> f32 {
        let mut value = self.bias;
        let previous_layer_len = previous_layer.len();
        if self.weights.len() != previous_layer_len {
            panic!("The number of weights ({}) doesn't match the number of values ({})", self.weights.len(), previous_layer_len);
        }
        for pos_num in 0..previous_layer_len {
            value += previous_layer[pos_num] * self.weights[pos_num];
        }
        value
    }
}

struct Layer {
    nodes: Vec<Node>,
}

impl Layer {
    pub fn calculate(&self, previous_layer:&Vec<f32>) -> Vec<f32> {
        let mut values = Vec::new();
        let node_count = self.nodes.len();
        for node_num in 0..node_count {
            values.push(self.nodes[node_num].calculate(previous_layer));
        }
        values
    }
}

fn model(app: &App) -> Model {
    let _window = app
    .new_window()
    .with_dimensions(700, 700)
    .with_title("Simple Neural Network")
    .view(view)
    .build()
    .unwrap();

    // The numbers 0-15 in binary
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

    // The numbers 0-15 as 7-segment display: "[1   8]" [B, C,     A, B, C, D, E, F, G]
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

    draw_results(model, &draw);

    draw.to_frame(app, frame).unwrap();
}


fn draw_results(model: &Model, draw: &nannou::app::Draw) {
    let t = model.time;

    let tdi = model.training_data_in[t];
    let tdu = model.training_data_out[t];

    draw.ellipse().x_y(-300.0, 5.0).radius(10.0).color(rgb(tdi[0], tdi[0], tdi[0])); // Input
    draw.ellipse().x_y(-300.0, 62.0).radius(10.0).color(rgb(tdi[1], tdi[1], tdi[1]));
    draw.ellipse().x_y(-300.0, 123.0).radius(10.0).color(rgb(tdi[2], tdi[2], tdi[2]));
    draw.ellipse().x_y(-300.0, 192.0).radius(10.0).color(rgb(tdi[3], tdi[3], tdi[3]));

    draw.ellipse().x_y(-180.0, 5.0).radius(6.0).color(rgb(1.0, 1.0, 1.0)); // Layer 1
    draw.ellipse().x_y(-180.0, 32.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 59.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 86.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 112.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 138.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 165.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 192.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));

    draw.ellipse().x_y(-180.0, 5.0).radius(6.0).color(rgb(1.0, 1.0, 1.0)); // Layer 2
    draw.ellipse().x_y(-180.0, 32.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 59.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 86.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 112.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 138.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 165.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-180.0, 192.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));

    draw.rect().x_y(50.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[0], tdu[0], tdu[0])); // B
    draw.rect().x_y(50.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[1], tdu[1], tdu[1])); // C

    draw.rect().x_y(135.0, 192.0).w_h(44.0, 16.0).color(rgb(tdu[2], tdu[2], tdu[2])); // A
    draw.tri().points((157.0, 200.0), (165.0, 200.0), (16.0, 0.0)).color(rgb(tdu[2], tdu[2], tdu[2])); // A corner
    draw.tri().points((113.0, 200.0), (105.0, 200.0), (113.0, 192.0)).color(rgb(tdu[2], tdu[2], tdu[2])); // A corner
    draw.rect().x_y(165.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[3], tdu[3], tdu[3])); // B
    draw.tri().points((157.0, 192.0), (173.0, 192.0), (172.0, 200.0)).color(rgb(tdu[3], tdu[3], tdu[3])); // B corner
    draw.rect().x_y(165.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[4], tdu[4], tdu[4])); // C
    draw.rect().x_y(135.0, 8.0).w_h(44.0, 16.0).color(rgb(tdu[5], tdu[5], tdu[5])); // D
    draw.rect().x_y(100.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[6], tdu[6], tdu[6])); // E
    draw.rect().x_y(100.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[7], tdu[7], tdu[7])); // F
    draw.rect().x_y(135.0, 100.0).w_h(44.0, 16.0).color(rgb(tdu[8], tdu[8], tdu[8])); // G
}
