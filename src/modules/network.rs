use crate::modules::{Neuron, NetworkInfo, Gene, NeatParams};
use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;
pub use bincode::{deserialize, serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Network {
  neurons: Vec<Neuron>,
}

impl Network {
  pub fn new(num_inputs: usize, num_outputs: usize, params: &NeatParams, network_info: &NetworkInfo, genes: &Vec<Gene>) -> Network {
    let mut neurons = network_info.get_input_neurons();
    
    let mut unique_nodes = Vec::new();
    for i in 0..num_outputs {
      unique_nodes.push(params.max_nodes()+i);
    }
    
    for i in 0..genes.len() {
      //if !genes[i].is_enabled() {
      //  continue;
      //}
      
      let out = genes[i].to();
      
      //if !unique_nodes.contains(&out) {
        unique_nodes.push(out);
      //}
    }
    
    for i in 0..num_outputs {
      let output_node_id = params.max_nodes()+i;
      //if !unique_nodes.contains(&output_node_id) {
        unique_nodes.push(output_node_id);
      //}
    }
    unique_nodes.dedup();
    neurons.append(&mut network_info.get_neurons_by_id(&unique_nodes));
    neurons.dedup();
    neurons.sort_by(|a, b| a.x().partial_cmp(&b.x()).unwrap());
    
    for i in 0..neurons.len() {
      let mut neuron = &mut neurons[i];
      
      for j in 0..genes.len() {
        let gene = &genes[j];
        
        if neuron.id() == gene.to() {
          let mut weight = gene.weight();
          if !gene.is_enabled() {
            weight = 0.0;
          }
          neuron.add_incoming(gene.from(), weight);
        }
      }
    }
    
    Network {
      neurons,
    }
  }
  
  pub fn neurons(&self) -> &Vec<Neuron> {
    &self.neurons
  }
  
  pub fn mut_neurons(&mut self) -> &mut Vec<Neuron> {
    &mut self.neurons
  }
  
  pub fn get_output_value(&self) -> Vec<f64> {
    let mut outputs = Vec::new();
    
    for i in 0..self.neurons.len() {
      if self.neurons[i].is_output() {
        outputs.push(self.neurons[i].value());
      }
    }
    
    outputs
  }
  
  pub fn get_neuron_by_id(&self, id: &usize) -> Option<Neuron> {
    let mut neuron_clone = None;
    for neuron in &self.neurons {
      if neuron.id() == *id {
         neuron_clone = Some(neuron.clone());
      }
    }
    
    neuron_clone
  }
  
  pub fn get_value_by_id(&self, id: &usize) -> f64 {
    let mut value = 0.0;
    for neuron in &self.neurons {
      if neuron.id() == *id {
        value = neuron.value();
      }
    }
    
    value
  }
  
  pub fn add_inputs(&mut self, inputs: Vec<f64>) {
    let mut offset = 0;
    let mut input_idx = 0;
    for i in 0..inputs.len() {
      for j in offset..self.neurons.len() {
        if self.neurons[j].is_input() {
          self.neurons[j].set_value(inputs[input_idx]);
          offset = j+1;
          input_idx += 1;
        }
      }
    }
  }
  
  pub fn reset_values(&mut self) {
    for i in 0..self.neurons.len() {
      self.neurons[i].set_value(0.0);
      self.neurons[i].set_sum(0.0);
    }
  }
  
  pub fn evaluate(&mut self) -> Vec<f64> {
    //print!("Expected Output: [");
    
    for i in 0..self.neurons.len() {
     /* if !self.neurons[i].is_input() {
        self.neurons[i].set_sum(0.0);
        self.neurons[i].set_value(0.0);
      }*/
      
      let mut should_activate = false;
      //println!("Neuron: {} v: {}", i, self.neurons[i].value());
      for j in 0..self.neurons[i].incoming().len() {
        let (from, weight) = self.neurons[i].incoming()[j];
        if weight == 0.0 {
          continue;
        }
        
        should_activate = true;
        let v = self.get_value_by_id(&from);
        
        let sum = self.neurons[i].sum();
        self.neurons[i].set_sum(sum + v*weight);
        
       // if self.neurons[i].is_output() {
          //println!("    Incoming: {}", j);
        //  println!("    from_value: {}", v);
         // println!("    neuron sum: {}", self.neurons[i].sum());
      //  }
      }
      //if self.neurons[i].is_output() {
        //let sum = self.neurons[i].sum();
        //print!("{}, ", sum);
       // self.neurons[i].set_value(sum);
      if should_activate && (self.neurons[i].is_output() || self.neurons[i].is_hidden()) {
       // println!("    Activated Neuron");
        //println!("      Before: {}", self.neurons[i].value());
        self.neurons[i].activate();
       // if self.neurons[i].is_output() {
       //   println!("      After: {}", self.neurons[i].value());
       // }
      }
    }
    //println!("] Actual Output {:?}", self.get_output_value());
    self.get_output_value()
  }
  
  pub fn random_neuron(&self, non_input: bool, rng: &mut ThreadRng) -> Neuron {
    let mut idx = 0;
    let mut neuron_not_found = true;
    
    let mut attempts = 0;
    
    while neuron_not_found {
      let rand = rng.gen_range(0..self.neurons.len());
      
      if self.neurons[rand].is_input() && !non_input || non_input {
        idx = rand;
        break;
      }
      
      if attempts > 2 {
        break;
      }
      attempts += 1;
    }
    
    self.neurons[idx].clone()
  }
  
  pub fn reposition_neurons(&mut self, rng: &mut ThreadRng) {
    let mut highest_x_pos = -1.0;
    let mut number_of_neurons = 0;
    let mut highest_idx = 0;
    
    // inputs are 0.1
    // outputs are 0.9
    
    self.neurons.sort_by(|a, b| a.x().partial_cmp(&b.x()).unwrap());
    let start = {
      let mut s = 0;
      for i in 0..self.neurons.len() { 
        if self.neurons[i].is_hidden() { s = i; break; } 
      }
      s
    };
    
    let end = {
      let mut e = 0;
      for i in 0..self.neurons.len() { 
        if self.neurons[i].is_output() { e = i-1; break; } 
      }
      e
    };
    
    let number_of_neurons = end-start;
    let segment_size = (0.9-0.1) / (number_of_neurons as f32+2.0);
    
    for i in start..end+1 {
      if self.neurons[i].is_hidden() {
        self.neurons[i].set_x(0.1 + segment_size*(i-start+1) as f32 + rng.gen::<f32>() * 0.01 - 0.005);
      }
    }
  }
}










