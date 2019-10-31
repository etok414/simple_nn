extern crate rand;

use rand::Rng;

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

    // pub fn find_adjusts(&mut self, previous_layer_values:&Vec<f32>, personal_value:f32, desired_value:f32, next_layer: &Layer) {
    // //Finds out how the node's bias and weights should be adjusted based on either a known desired value or how the next layer is set to be adjusted.
    // //If it receives a next_layer with any nodes in it, it uses the bias_adjusts of the nodes in that layer that connect with it along with its personal_value to find its own bias_adjust.
    // //Otherwise, it finds its bias_adjust from its personal_value and a specified desired_value.
    // //It then finds its weight_adjusts from its bias_adjust and previous_layer_values.
    //     if self.bias_adjust.is_some() {
    //         panic!("The bias adjust contained a value before adjusts were found");
    //     }
    //     if self.weight_adjusts.len() != 0 {
    //         panic!("The weight adjusts contained {} value(s) before adjusts were found", self.weight_adjusts.len());
    //     }
    //     let previous_layer_len = previous_layer_values.len();
    //     if self.weights.len() != previous_layer_len {
    //         panic!("The number of weights ({}) doesn't match the number of values ({})", self.weights.len(), previous_layer_len);
    //     }
    //     let mut delta = 0.0;
    //     if next_layer.node_count == 0 {
    //         delta = (personal_value - desired_value) * personal_value * (1.0 - personal_value)
    //     } else {
    //         for num in 0..next_layer.node_count {
    //             delta += next_layer.nodes[num].bias_adjust.unwrap() * next_layer.nodes[num].weights[desired_value as usize]; //If there is a next_layer, desired_value tracks which connections in it lead to the node.
    //         }
    //         delta *= personal_value * (1.0 - personal_value);
    //     }
    //     self.bias_adjust = Some(delta);
    //     for pos_num in 0..previous_layer_len {
    //         self.weight_adjusts.push(delta * previous_layer_values[pos_num]);
    //     }
    // }
    //
    // pub fn adjust(&mut self, learning_rate: f32) {
    // //Changes the weights and biases after all nodes have found how they're supposed to be adjusted.
    //     if self.bias_adjust.is_none() {
    //         panic!{"Bias adjust is missing"}
    //     }
    //     let weight_count = self.weights.len();
    //     if self.weight_adjusts.len() != weight_count {
    //         panic!("The number of weight adjusts ({}) doesn't match the number of weights ({})", self.weight_adjusts.len(), weight_count);
    //     }
    //     self.bias -= self.bias_adjust.unwrap() * learning_rate;
    //     self.bias_adjust = None;
    //     for num in 0..weight_count {
    //         self.weights[num] -= self.weight_adjusts[num] * learning_rate;
    //     }
    //     self.weight_adjusts = Vec::new();
    // }

    pub fn find_delta(&self, personal_value:f32, desired_value:f32, next_layer: &Layer, next_layer_deltas: &Vec<f32>) -> f32 {
        //Finds delta and returns it.
        let mut delta = 0.0;
        if next_layer.node_count == 0 {
            delta = (personal_value - desired_value) * personal_value * (1.0 - personal_value)
        } else {
            for num in 0..next_layer.node_count {
                delta += next_layer_deltas[num] * next_layer.nodes[num].weights[desired_value as usize]; //If there is a next_layer, desired_value tracks which connections in it lead to the node.
            }
            delta *= personal_value * (1.0 - personal_value);
        }
        delta
    }

    // pub fn single_adjust(&mut self, delta: f32, learning_rate: f32, previous_layer_value: f32, relevant_weight: usize) {
    //     if relevant_weight == self.weights.len() {
    //         self.bias -= delta * learning_rate;
    //     } else {
    //         self.weights[relevant_weight] -= delta * previous_layer_value * learning_rate;
    //     }
    // }
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

    // pub fn find_adjusts(&mut self, previous_layer_values:&Vec<f32>, values:&Vec<f32>, desired_values:&Vec<f32>, next_layer: &Layer) {
    // //Finds out how the nodes' weights and biases should be adjusted, based either on a list of desired values, or how the next layer is set to be adjusted.
    // //The function does this by calling find_adjusts for each node.
    //     for node_num in 0..self.node_count {
    //         self.nodes[node_num].find_adjusts(
    //                                           previous_layer_values,
    //                                           values[node_num], //personal_value, is used to track the calculated value of the relevant node.
    //                                           if next_layer.node_count > 0 {node_num as f32} //If the next layer contains anything, desired_value is used to track the position of the relevant node.
    //                                             else {desired_values[node_num]}, //Otherwise, desired_value is used to track the desired value.
    //                                           next_layer
    //                                           );
    //     }
    // }
    //
    // pub fn adjust(&mut self, learning_rate:f32) {
    // //Calls the adjust function of every node in the layer, which does the following for the node:
    // //Changes the weights and biases after all nodes have found how they're supposed to be adjusted.
    //     for node_num in 0..self.node_count {
    //         self.nodes[node_num].adjust(learning_rate)
    //     }
    // }

    pub fn find_deltas(&self, values:&Vec<f32>, desired_values:&Vec<f32>, next_layer:&Layer, next_layer_deltas:&Vec<f32>) -> Vec<f32> {
    //Finds out how the nodes' weights and biases should be adjusted, based either on a list of desired values, or how the next layer is set to be adjusted.
    //The function does this by calling find_adjusts for each node.
        let deltas = Vec::new();
        for node_num in 0..self.node_count {
            deltas.push(self.nodes[node_num].find_delta(
                                                        values[node_num],
                                                        if next_layer.node_count > 0 {node_num as f32} //If the next layer contains anything, desired_value is used to track the position of the relevant node.
                                                            else {desired_values[node_num]}, //Otherwise, desired_value is used to track the desired value.
                                                        next_layer,
                                                        next_layer_deltas
                                                        )
                        );
        }
        deltas
    }

    pub fn alt_adjust(&mut self, deltas: Vec<f32>, previous_layer_values: Vec<f32>, learning_rate: f32) {
        let previous_layer_len = previous_layer_values.len();
        for prev_num in 0..previous_layer_len {
            for num in 0..self.node_count {
                self.nodes[num].weights[prev_num] -= deltas[num] * previous_layer_values[prev_num] * learning_rate;
            }
        }
        for num in 0..self.node_count {
            self.nodes[num].bias -= deltas[num] * learning_rate;
        }
    }
}

struct Network {
    layers: Vec<Layer>,
    layer_count: usize,
    learning_rate: f32,
}

impl Network {
    pub fn new(node_nums:Vec<usize>, learning_rate: f32) -> Network {
        let mut layers = Vec::new();
        for layer_num in 1..node_nums.len() {
            layers.push(Layer::new(layer_num-1, layer_num));
        }
        let layer_count = layers.len();
        Network {
            layers: layers,
            layer_count: layer_count,
            learning_rate: learning_rate,
        }
    }

    pub fn calculate(&self, inputs: &Vec<f32>) -> Vec<Vec<f32>> {
    //Calculates the values of all nodes based on the active training data and the weights and biases.
    //The outer vector of the output is the layer, the inner vector is the position in the layer. To get the output layer from values, say values[values.len() - 1]
        let mut values = vec![self.layers[0].calculate(inputs)];
        for num in 1..self.layer_count {
            values.push(self.layers[num].calculate(&values[num-1]));
        }
        values
    }

    // pub fn find_make_adjust(&mut self, inputs: &Vec<f32>, desired_outputs:&Vec<f32>) {
    // //Finds out how the weights and biases should be adjusted, based on the difference between the results of the calculate function and the desired outputs.
    // //It does it in reverse order, because that's how you have to do it.
    // //Then it implements all of the changes after they have all been calculated. That doesn't have to be done in reverse order, so it isn't.
    //     let values = self.calculate(inputs);
    //     self.layers[self.layer_count-1].find_adjusts(&values[self.layer_count-2], &values[self.layer_count-1], desired_outputs, &Layer::new(0, 0));
    //     for num in (1..self.layer_count-1).rev() {
    //         let next_layer = self.layers[num+1].clone();
    //         self.layers[num].find_adjusts(&values[num-1], &values[num], &Vec::new(), &next_layer);
    //     }
    //     let next_layer = self.layers[1].clone();
    //     self.layers[0].find_adjusts(inputs, &values[0], &Vec::new(), &next_layer);
    //
    //     for num in 0..self.layer_count {
    //         self.layers[num].adjust(self.learning_rate);
    //     }
    // }

    pub fn alt_find_make_adjust(&mut self, inputs: &Vec<f32>, desired_outputs:&Vec<f32>) {
        let mut delta_matrix = vec![Vec::new(); self.layer_count];
        delta_matrix[self.layer_count-1] = self.layers[self.layer_count-1].find_deltas(&values[self.layer_count-1], desired_outputs, &Layer::new(0, 0), Vec::new());
        for num in (0..self.layer_count-1).rev() {
            delta_matrix[num] = self.layers[num].find_deltas(&values[num], &Vec::new(), self.layers[num+1], &delta_matrix[num+1]);
        }

        let values = self.calculate(inputs);
        self.layers[0].alt_adjust(delta_matrix[0], inputs, self.learning_rate)
        for num in 1..self.layer_count {
            self.layers[num].alt_adjust(delta_matrix[num], values[num-1], self.learning_rate)
        }
    }
}
