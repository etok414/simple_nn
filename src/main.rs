extern crate rand;

use nannou::prelude::*;
use std::thread::sleep;
use std::time;
use rand::Rng;


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
    bias_adjust: Option<f32>,
    weight_adjusts: Vec<f32>,
}

impl Node {
    pub fn new(number_of_weights: usize) -> Node {
        let mut rng = rand::thread_rng();

        let mut init_weights = vec![0.0; number_of_weights];

        for n in 0..number_of_weights {
            let x: f32 = rng.gen();  // Random number in the interval [0; 1[
            init_weights[n] = 2.0 * x  - 1.0;  // The initial weights will be in [-1; 1[
        }

        Node {bias: 0.0,
              weights: init_weights, // vec![0.0; number_of_weights],
              bias_adjust: None,
              weight_adjusts: Vec::new()}
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
        let norm_value = 1.0 / (1.0 + (-value).exp());
        if norm_value < 0.0 || norm_value > 1.0 {
            panic!{"Math is broken, the sigmoid functions returns value outside [0; 1]"}
        }
        norm_value
    }

    pub fn find_adjusts(&mut self, previous_layer:&Vec<f32>, personal_value:f32, desired_value:f32, next_layer: &Layer) {
        if self.bias_adjust.is_some() {
            panic!("The bias adjust contained a value before adjusts were found");
        }
        if self.weight_adjusts.len() != 0 {
            panic!("The weight adjusts contained {} value(s) before adjusts were found", self.weight_adjusts.len());
        }
        let previous_layer_len = previous_layer.len();
        if self.weights.len() != previous_layer_len {
            panic!("The number of weights ({}) doesn't match the number of values ({})", self.weights.len(), previous_layer_len);
        }
        let mut delta = 0.0;
        if next_layer.nodes.len() == 0 {
            delta = (personal_value - desired_value) * personal_value * (1.0 - personal_value)
        } else {
            let next_len = next_layer.nodes.len();
            for num in 0..next_len {
                delta += next_layer.nodes[num].bias_adjust.unwrap();
            }
            delta *= personal_value * (1.0 - personal_value);
        }
        self.bias_adjust = Some(delta);
        for pos_num in 0..previous_layer_len {
            self.weight_adjusts.push(delta * previous_layer[pos_num]);
        }
    }

    // fn find_weight_adjust(previous_node:f32, personal_value:f32, desired_value:f32) -> f32 {
    //     (personal_value - desired_value) * personal_value * (1.0 + personal_value) * previous_node
    // }

    pub fn adjust(&mut self, learning_rate: f32) {
        if self.bias_adjust.is_none() {
            panic!{"Bias adjust is missing"}
        }
        let weight_count = self.weights.len();
        if self.weight_adjusts.len() != weight_count {
            panic!("The number of weight adjusts ({}) doesn't match the number of weights ({})", self.weight_adjusts.len(), weight_count);
        }
        self.bias -= self.bias_adjust.unwrap() * learning_rate;
        self.bias_adjust = None;
        for num in 0..weight_count {
            self.weights[num] -= self.weight_adjusts[num] * learning_rate;
        }
        self.weight_adjusts = Vec::new();
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

    pub fn find_adjusts(&mut self, previous_layer:&Vec<f32>, values:&Vec<f32>, desired_values:&Vec<f32>, next_layer: &Layer) {
        let node_count = self.nodes.len();
        for node_num in 0..node_count {
            self.nodes[node_num].find_adjusts(previous_layer, values[node_num], desired_values[node_num], next_layer);
        }
    }

    pub fn adjust(&mut self, learning_rate:f32) {
        let node_count = self.nodes.len();
        for node_num in 0..node_count {
            self.nodes[node_num].adjust(learning_rate)
        }
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

    draw.ellipse().x_y(-80.0, 5.0).radius(6.0).color(rgb(1.0, 1.0, 1.0)); // Layer 2
    draw.ellipse().x_y(-80.0, 32.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-80.0, 59.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-80.0, 86.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-80.0, 112.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-80.0, 138.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-80.0, 165.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));
    draw.ellipse().x_y(-80.0, 192.0).radius(6.0).color(rgb(1.0, 1.0, 1.0));

    // 1 part of the 7 segment display
    draw.rect().x_y(50.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[1], tdu[1], tdu[1])); // B
    draw.tri().points((42.0, 192.0), (58.0, 192.0), (58.0, 200.0)).color(rgb(tdu[3], tdu[3], tdu[3])); // B corner
    draw.tri().points((42.0, 108.0), (58.0, 108.0), (58.0, 100.0)).color(rgb(tdu[3], tdu[3], tdu[3])); // B corner
    draw.rect().x_y(50.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[0], tdu[0], tdu[0])); // C

    // 8 part of the 7 segment display
    draw.rect().x_y(135.0, 192.0).w_h(44.0, 16.0).color(rgb(tdu[2], tdu[2], tdu[2])); // A
    draw.tri().points((157.0, 200.0), (165.0, 200.0), (157.0, 184.0)).color(rgb(tdu[2], tdu[2], tdu[2])); // A corner
    draw.tri().points((113.0, 200.0), (105.0, 200.0), (113.0, 184.0)).color(rgb(tdu[2], tdu[2], tdu[2])); // A corner
    draw.rect().x_y(165.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[3], tdu[3], tdu[3])); // B
    draw.tri().points((157.0, 192.0), (173.0, 192.0), (173.0, 200.0)).color(rgb(tdu[3], tdu[3], tdu[3])); // B corner
    draw.tri().points((157.0, 108.0), (173.0, 108.0), (173.0, 100.0)).color(rgb(tdu[3], tdu[3], tdu[3])); // B corner
    draw.rect().x_y(165.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[4], tdu[4], tdu[4])); // C
    draw.rect().x_y(135.0, 8.0).w_h(44.0, 16.0).color(rgb(tdu[5], tdu[5], tdu[5])); // D
    draw.rect().x_y(100.0, 50.0).w_h(16.0, 84.0).color(rgb(tdu[6], tdu[6], tdu[6])); // E
    draw.rect().x_y(100.0, 150.0).w_h(16.0, 84.0).color(rgb(tdu[7], tdu[7], tdu[7])); // F
    draw.rect().x_y(135.0, 100.0).w_h(44.0, 16.0).color(rgb(tdu[8], tdu[8], tdu[8])); // G
}
