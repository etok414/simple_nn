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
    relevant_data: usize,
    learning_rate: f32,
    layers: Vec<Layer>,
    _window: WindowId,
}

#[derive(Clone)]
struct Node {
// A node/neuron's bias and the weights of its connections to the previous layer.
    bias: f32,
    weights: Vec<f32>,
    bias_adjust: Option<f32>,
    weight_adjusts: Vec<f32>,
}

impl Node {
    pub fn new(number_of_weights: usize) -> Node {
    //Generates a new node with a random bias and random weights.
        let mut rng = rand::thread_rng();

        let mut init_weights = vec![0.0; number_of_weights];

        for n in 0..number_of_weights {
            let x: f32 = rng.gen();  // Random number in the interval [0; 1[
            init_weights[n] = 2.0 * x  - 1.0;  // The initial weights will be in [-1; 1[
        }
        let x: f32 = rng.gen();  // Random number in the interval [0; 1[
        let bias = 2.0 * x  - 1.0;  // The initial weights will be in [-1; 1[

        Node {
            bias: bias,
            weights: init_weights, // vec![0.0; number_of_weights],
            bias_adjust: None,
            weight_adjusts: Vec::new()
        }
    }

    pub fn calculate(&self, previous_layer_values:&Vec<f32>) -> f32 {
    //Calculates the value of the node based on the values of the previous layer and the node's bias and weights.
        let mut value = self.bias;
        let previous_layer_len = previous_layer_values.len();
        if self.weights.len() != previous_layer_len {
            panic!("The number of weights ({}) doesn't match the number of values ({})", self.weights.len(), previous_layer_len);
        }
        for pos_num in 0..previous_layer_len {
            value += previous_layer_values[pos_num] * self.weights[pos_num];
        }
        let norm_value = 1.0 / (1.0 + (-value).exp());
        if norm_value < 0.0 || norm_value > 1.0 {
            panic!{"Math is broken, the sigmoid functions returns value outside [0; 1]"}
        }
        norm_value
    }

    pub fn find_adjusts(&mut self, previous_layer_values:&Vec<f32>, personal_value:f32, desired_value:f32, next_layer: &Layer) {
    //Finds out how the node's bias and weights should be adjusted based on either a known desired value or how the next layer is set to be adjusted.
    //If it receives a next_layer with any nodes in it, it uses the bias_adjusts of the nodes in that layer that connect with it along with its personal_value to find its own bias_adjust.
    //Otherwise, it finds its bias_adjust from its personal_value and a specified desired_value.
    //It then finds its weight_adjusts from its bias_adjust and previous_layer_values.
        if self.bias_adjust.is_some() {
            panic!("The bias adjust contained a value before adjusts were found");
        }
        if self.weight_adjusts.len() != 0 {
            panic!("The weight adjusts contained {} value(s) before adjusts were found", self.weight_adjusts.len());
        }
        let previous_layer_len = previous_layer_values.len();
        if self.weights.len() != previous_layer_len {
            panic!("The number of weights ({}) doesn't match the number of values ({})", self.weights.len(), previous_layer_len);
        }
        let mut delta = 0.0;
        if next_layer.node_count == 0 {
            delta = (personal_value - desired_value) * personal_value * (1.0 - personal_value)
        } else {
            for num in 0..next_layer.node_count {
                delta += next_layer.nodes[num].bias_adjust.unwrap() * next_layer.nodes[num].weights[desired_value as usize]; //If there is a next_layer, desired_value tracks which connections in it lead to the node.
            }
            delta *= personal_value * (1.0 - personal_value);
        }
        self.bias_adjust = Some(delta);
        for pos_num in 0..previous_layer_len {
            self.weight_adjusts.push(delta * previous_layer_values[pos_num]);
        }
    }

    // fn find_weight_adjust(previous_node:f32, personal_value:f32, desired_value:f32) -> f32 {
    //     (personal_value - desired_value) * personal_value * (1.0 + personal_value) * previous_node
    // }

    pub fn adjust(&mut self, learning_rate: f32) {
    //Changes the weights and biases after all nodes have found how they're supposed to be adjusted.
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

#[derive(Clone)]
struct Layer {
//A layer of nodes, their biases, and the weights of their connections to the previous layer.
    nodes: Vec<Node>,
    node_count: usize, //Should be equal to nodes.len() and shouldn't change.
}

impl Layer {
    pub fn new(previous_layer_nodes: usize, number_of_nodes: usize) -> Layer {
    //Generates a layer of nodes, each with a random bias and a number of random weights equal to the number of nodes in the previous layer.
        let mut nodes = Vec::new();
        for _ in 0..number_of_nodes {
            nodes.push(Node::new(previous_layer_nodes));
        }
        let node_count = nodes.len();
        Layer {
            nodes: nodes,
            node_count: node_count,
        }
    }

    pub fn calculate(&self, previous_layer_values:&Vec<f32>) -> Vec<f32> {
    //Calculates the values of the nodes based on the values of the previous layer and the nodes' weights and biases.
        let mut values = Vec::new();
        for node_num in 0..self.node_count {
            values.push(self.nodes[node_num].calculate(previous_layer_values));
        }
        values
    }

    pub fn find_adjusts(&mut self, previous_layer_values:&Vec<f32>, values:&Vec<f32>, desired_values:&Vec<f32>, next_layer: &Layer) {
    //Finds out how the nodes' weights and biases should be adjusted, based either on a list of desired values, or how the next layer is set to be adjusted.
    //The function does this by calling find_adjusts for each node.
        for node_num in 0..self.node_count {
            self.nodes[node_num].find_adjusts(
                                              previous_layer_values,
                                              values[node_num], //personal_value, is used to track the calculated value of the relevant node.
                                              if next_layer.node_count > 0 {node_num as f32} //If the next layer contains anything, desired_value is used to track the position of the relevant node.
                                                else {desired_values[node_num]}, //Otherwise, desired_value is used to track the desired value.
                                              next_layer
                                              );
        }
    }

    pub fn adjust(&mut self, learning_rate:f32) {
    //Calls the adjust function of every node in the layer, which does the following for the node:
    //Changes the weights and biases after all nodes have found how they're supposed to be adjusted.
        for node_num in 0..self.node_count {
            self.nodes[node_num].adjust(learning_rate)
        }
    }
}

// fn calculate(inputs: &Vec<f32>, layers:&Vec<Layer>) -> Vec<Vec<f32>> {
fn calculate(model: &Model) -> Vec<Vec<f32>> {
//Calculates the values of all nodes based on the active training data and the weights and biases.
//The outer vector of the output is the layer, the inner vector is the position in the layer. To get the output layer from values, say values[values.len() - 1]
    let mut values = vec![model.layers[0].calculate(&model.training_data_in[model.relevant_data].to_vec())];
    for num in 1..model.layers.len() {
        values.push(model.layers[num].calculate(&values[num-1]));
    }
    values
}

fn find_cost(model: &Model) -> f32 {
//Finds the cost function of the active training data, which is the difference between the current result and the desired result.
//Not actually used for anything, since the find_adjust use calculations that have already taken the cost function into accout.
    let values = calculate(model);
    let mut cost = 0.0;
    for (num, value) in values[values.len() - 1].iter().enumerate() {
        cost += (value - model.training_data_out[model.relevant_data][num]).powi(2)/2.0;
    }
    cost
}

// fn find_make_adjust(inputs: &Vec<f32>, layers:&mut Vec<Layer>, desired_outputs:&Vec<f32>, learning_rate:f32) {
fn find_make_adjust(model: &mut Model) {
//Finds out how the weights and biases should be adjusted, based on the difference between the results of the calculate function and the desired outputs.
//It does it in reverse order, because that's how you have to do it.
//Then it implements all of the changes after they have all been calculated. That doesn't have to be done in reverse order, so it isn't.
    let values = calculate(&model);
    let layer_count = model.layers.len();
    model.layers[layer_count-1].find_adjusts(&values[layer_count-2], &values[layer_count-1], &model.training_data_out[model.relevant_data].to_vec(), &Layer::new(0, 0));
    for num in (1..layer_count-1).rev() {
        let next_layer = model.layers[num+1].clone();
        model.layers[num].find_adjusts(&values[num-1], &values[num], &Vec::new(), &next_layer);
    }
    let next_layer = model.layers[1].clone();
    model.layers[0].find_adjusts(&model.training_data_in[model.relevant_data].to_vec(), &values[0], &Vec::new(), &next_layer);
    for num in 0..layer_count {
        model.layers[num].adjust(model.learning_rate);
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
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], // 0
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], // 5
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], //10
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], // 15
    ] ;

    let time = 0;

    let relevant_data = 1;

    let learning_rate = 0.5;

    let mut layers = Vec::new();
    layers.push(Layer::new(4, 8));
    layers.push(Layer::new(8, 8));
    layers.push(Layer::new(8, 9));

    Model {
        training_data_in,
        training_data_out,
        time,
        relevant_data,
        learning_rate,
        layers,
        _window }
}


fn update(_app: &App, model: &mut Model, _update: Update) {

    find_make_adjust(model);
    model.time += 1;
    model.relevant_data += 1;

    if model.relevant_data > 15 {
        model.relevant_data = 0;
    }

    // if model.time > 4_000 {
    if model.time % 512 < 16 {
        println!("time: {:?} cost: {:?}", model.time, find_cost(model));
        sleep(time::Duration::new(0, 500000000)); // sec, nano sec
    }
}


fn view(app: &App, model: &Model, frame: &Frame) {

    let draw = app.draw();

    draw_results(model, &draw);

    draw.to_frame(app, frame).unwrap();
}


fn draw_results(model: &Model, draw: &nannou::app::Draw) {

    let tdi = model.training_data_in[model.relevant_data];
    // let tdu = model.training_data_out[model.relevant_data];
    let results = calculate(model);
    let tdu = &results[results.len() - 1];

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

    // "1" part of the 7 segment display
    draw.polygon().color(rgb(tdu[1], tdu[1], tdu[1])).points(vec!(pt2(106.0 - 62.0, 184.0), pt2(112.0 - 62.0, 178.0),pt2(112.0 - 62.0, 112.0), pt2(106.0 - 62.0, 103.0), pt2(96.0 - 62.0, 109.0), pt2(96.0 - 62.0, 176.0))); // B
    draw.polygon().color(rgb(tdu[0], tdu[0], tdu[0])).points(vec!(pt2(106.0 - 62.0, 184.0 - 85.0), pt2(112.0 - 62.0, 178.0 - 85.0),pt2(112.0 - 62.0, 112.0 - 85.0), pt2(106.0 - 62.0, 103.0 - 85.0), pt2(96.0 - 62.0, 109.0 - 85.0), pt2(96.0 - 62.0, 176.0 - 85.0))); // C

    // "8" part of the 7 segment display
    draw.polygon().color(rgb(tdu[2], tdu[2], tdu[2])).points(vec!(pt2(108.0, 190.0), pt2(113.0, 197.0),pt2(163.0, 197.0), pt2(172.0, 189.0), pt2(161.0, 182.0), pt2(115.0, 182.0))); // A
    draw.polygon().color(rgb(tdu[3], tdu[3], tdu[3])).points(vec!(pt2(106.0 + 67.0, 184.0), pt2(112.0 + 67.0, 178.0),pt2(112.0 + 67.0, 112.0), pt2(106.0 + 67.0, 103.0), pt2(96.0 + 67.0, 109.0), pt2(96.0 + 67.0, 176.0))); // B
    draw.polygon().color(rgb(tdu[4], tdu[4], tdu[4])).points(vec!(pt2(106.0 + 67.0, 184.0 - 85.0), pt2(112.0 + 67.0, 178.0 - 85.0),pt2(112.0 + 67.0, 112.0 - 85.0), pt2(106.0 + 67.0, 103.0 - 85.0), pt2(96.0 + 67.0, 109.0 - 85.0), pt2(96.0 + 67.0, 176.0 - 85.0))); // C
    draw.polygon().color(rgb(tdu[5], tdu[5], tdu[5])).points(vec!(pt2(108.0, 190.0 - 176.0), pt2(113.0, 197.0 - 176.0),pt2(163.0, 197.0 - 176.0), pt2(172.0, 189.0 - 176.0), pt2(161.0, 182.0 - 176.0), pt2(115.0, 182.0 - 176.0))); // D
    draw.polygon().color(rgb(tdu[6], tdu[6], tdu[6])).points(vec!(pt2(106.0, 184.0 - 85.0), pt2(112.0, 178.0 - 85.0),pt2(112.0, 112.0 - 85.0), pt2(106.0, 103.0 - 85.0), pt2(96.0, 109.0 - 85.0), pt2(96.0, 176.0 - 85.0))); // E
    draw.polygon().color(rgb(tdu[7], tdu[7], tdu[7])).points(vec!(pt2(106.0, 184.0), pt2(112.0, 178.0),pt2(112.0, 112.0), pt2(106.0, 103.0), pt2(96.0, 109.0), pt2(96.0, 176.0))); // F
    draw.polygon().color(rgb(tdu[8], tdu[8], tdu[8])).points(vec!(pt2(108.0, 190.0 - 90.0), pt2(113.0, 197.0 - 90.0),pt2(163.0, 197.0 - 90.0), pt2(172.0, 189.0 - 90.0), pt2(161.0, 182.0 - 90.0), pt2(115.0, 182.0 - 90.0))); // G
}
