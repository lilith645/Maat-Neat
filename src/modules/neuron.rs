use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;
pub use bincode::{deserialize, serialize};

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum NodeType {
  Input,
  Hidden,
  Output,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Neuron {
  incoming: Vec<(usize, f64)>,
  value: f64,
  id: usize,
  x: f32,
  y: f32,
  sum: f64,
  node_type: NodeType,
}

impl Neuron {
  pub fn new(node_type: NodeType, x: f32, y: f32, id: usize) -> Neuron {
    Neuron {
      incoming: Vec::new(),
      value: 0.0,
      id,
      x,
      y,
      sum: 0.0,
      node_type,
    }
  }
  
  fn sigmoid(x: f64) -> f64 {
    2.0 / (1.0 + (-4.9 * x).exp())-1.0
    //1.0 / (1.0 + (-x).exp())
    //x.tanh()
  }
  
  pub fn id(&self) -> usize {
    self.id
  }
  
  pub fn node_type(&self) -> NodeType {
    self.node_type
  }
  
  pub fn x(&self) -> f32 {
    self.x
  }
  
  pub fn y(&self) -> f32 {
    self.y
  }
  
  pub fn sum(&self) -> f64 {
    self.sum
  }
  
  pub fn value(&self) -> f64 {
    self.value
  }
  
  pub fn is_input(&self) -> bool {
    self.node_type == NodeType::Input
  }
  
  pub fn is_hidden(&self) -> bool {
    self.node_type == NodeType::Hidden
  }
  
  pub fn is_output(&self) -> bool {
    self.node_type == NodeType::Output
  }
  
  pub fn incoming(&self) -> &Vec<(usize, f64)> {
    &self.incoming
  }
  
  pub fn set_x(&mut self, x: f32) {
    self.x = x;
  }
  
  pub fn set_y(&mut self, y: f32) {
    self.y = y;
  }
  
  pub fn add_incoming(&mut self, new_in: usize, weight: f64) {
    if !self.incoming.contains(&(new_in, weight)) {
      self.incoming.push((new_in, weight));
    }
  }
  
  pub fn set_value(&mut self, v: f64) {
    self.value = v;
  }
  
  pub fn set_sum(&mut self, v: f64) {
    self.sum = v;
  }
  /*
  pub fn add_in_value(&mut self, from: usize, value: f64) {
    for (node, weight) in &self.incoming {
      if *node == from {
        self.sum += value * weight;
        break;
      }
    }
  }*/
  
  pub fn activate(&mut self) {
    self.value = Neuron::sigmoid(self.sum);
  }
}






