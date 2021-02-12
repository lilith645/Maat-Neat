use crate::modules::{Neuron, Gene, NodeType};
use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;
pub use bincode::{deserialize, serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NeatParams {
  delta_disjoint: f64,
  delta_weights: f64,
  delta_threshold: f64,
  stale_species: u32,
  mutate_connection_chance: f64,
  perturb_chance: f64,
  crossover_chance: f64,
  link_mutation_chance: f64,
  node_mutation_chance: f64,
  bias_mutation_chance: f64,
  step_size: f64,
  disable_mutation_chance: f64,
  enable_mutation_chance: f64,
  target_species: usize,
  max_nodes: usize,
}

impl NeatParams {
  pub fn new() -> NeatParams {
    NeatParams {
      delta_disjoint: 2.0,
      delta_weights: 0.4,
      delta_threshold: 1.0,
      stale_species: 15,
      mutate_connection_chance: 0.25,
      perturb_chance: 0.9,
      crossover_chance: 0.75,
      link_mutation_chance: 2.0,
      node_mutation_chance: 0.5,
      bias_mutation_chance: 0.4,
      step_size: 0.1,
      disable_mutation_chance: 0.4,
      enable_mutation_chance: 0.2,
      target_species: 4,
      max_nodes: 1000000,
    }
  }
  
  pub fn new_arkanoid() -> NeatParams {
    NeatParams {
      delta_disjoint: 2.0,
      delta_weights: 0.8,
      delta_threshold: 1.0,
      stale_species: 15,
      mutate_connection_chance: 2.5,
      perturb_chance: 9.0,
      crossover_chance: 7.5,
      link_mutation_chance: 20.0,
      node_mutation_chance: 5.0,
      bias_mutation_chance: 4.0,
      step_size: 0.1,
      disable_mutation_chance: 4.0,
      enable_mutation_chance: 2.0,
      target_species: 4,
      max_nodes: 1000000,
    }
  }
  
  pub fn multiply_mutation_rates(&mut self, x: f64) {
    //self.delta_disjoint *= x;
   // self.delta_weights *= x;
    //self.delta_threshold *= x;
    //self.stale_species *= x;
    self.mutate_connection_chance *= x;
    self.perturb_chance *= x;
    self.crossover_chance *= x;
    self.link_mutation_chance *= x;
    self.node_mutation_chance *= x;
    self.bias_mutation_chance *= x;
    self.step_size *= x;
    self.disable_mutation_chance *= x;
    self.enable_mutation_chance *= x;
  }
  
  pub fn max_nodes(&self) -> usize {
    self.max_nodes
  }
  
  pub fn delta_disjoint(&self) -> f64 {
    self.delta_disjoint
  }
  
  pub fn delta_weights(&self) -> f64 {
    self.delta_weights
  }
  
  pub fn delta_threshold(&self) -> f64 {
    self.delta_threshold
  }
  
  pub fn stale_species(&self) -> u32 {
    self.stale_species
  }
  
  pub fn mutate_connection_chance(&self) -> f64 {
    self.mutate_connection_chance
  }
  
  pub fn perturb_chance(&self) -> f64 {
    self.perturb_chance
  }
  
  pub fn crossover_chance(&self) -> f64 {
    self.crossover_chance
  }
  
  pub fn link_mutation_chance(&self) -> f64 {
    self.link_mutation_chance
  }
  
  pub fn node_mutation_chance(&self) -> f64 {
    self.node_mutation_chance
  }
  
  pub fn bias_mutation_chance(&self) -> f64 {
    self.bias_mutation_chance
  }
  
  pub fn step_size(&self) -> f64 {
    self.step_size
  }
  
  pub fn disable_mutation_chance(&self) -> f64 {
    self.disable_mutation_chance
  }
  
  pub fn enable_mutation_chance(&self) -> f64 {
    self.enable_mutation_chance
  }
  
  pub fn target_species(&self) -> usize {
    self.target_species
  }
  
  pub fn decrease_delta_threshold(&mut self) {
    self.delta_threshold -= 0.3;
    if self.delta_threshold <= 0.3 {
      self.delta_threshold = 0.3;
    }
  }
  
  pub fn increase_delta_threshold(&mut self) {
    self.delta_threshold += 0.3;
  }
}

pub struct NetworkInfo {
  num_inputs: usize,
  num_outputs: usize,
  genes: Vec<Gene>,
  neurons: Vec<Neuron>,
  innovation: usize,
  last_id: usize,
}

impl NetworkInfo {
  pub fn new_empty() -> NetworkInfo {
    NetworkInfo {
      num_inputs: 0,
      num_outputs: 0,
      genes: Vec::new(),
      neurons: Vec::new(),
      innovation: 0,
      last_id: 0,
    }
  }
  
  pub fn new(num_inputs: usize, num_outputs: usize, params: &NeatParams) -> NetworkInfo {
    let mut neurons = Vec::new();
    let mut last_id = 0;
    
    let columns = (num_inputs as f32).sqrt().ceil() as usize;
    
    for i in 0..num_inputs+1 { // include bias
      let size = 0.4;
      let mut x = ((i/columns + 1) as f32 / (columns + 2) as f32)*size;
      let mut y = ((i%columns + 1) as f32 / (columns + 2) as f32);
      
      neurons.push(Neuron::new(NodeType::Input, 
                               0.1 - if i == num_inputs { 0.05 } else { 0.0 }, 
                               (i + 1) as f32 / (num_inputs + 2) as f32,
                               i))
    }
    
    for i in 0..num_outputs {
      neurons.push(Neuron::new(NodeType::Output, 
                               0.9, (i + 1) as f32 / (num_outputs + 1) as f32,
                               params.max_nodes() as usize+i));
      last_id = params.max_nodes() as usize+i;
    }
    
    NetworkInfo {
      num_inputs,
      num_outputs,
      genes: Vec::new(),
      neurons,
      innovation: 0,
      last_id: num_inputs+1,
    }
  }
  
  pub fn innovations(&self) -> usize {
    self.innovation
  }
  
  pub fn neurons(&self) -> &Vec<Neuron> {
    &self.neurons
  }
  
  pub fn get_input_neurons(&self) -> Vec<Neuron> {
    let mut n = Vec::new();
    
    for i in 0..self.num_inputs+1 {
      n.push(self.neurons[i].clone());
    }
    
    n
  }
  
  pub fn get_output_neurons(&self) -> Vec<Neuron> {
    let mut n = Vec::new();
    
    for i in 0..self.num_outputs {
      n.push(self.neurons[self.num_inputs+i].clone());
    }
    
    n
  }
  
  pub fn get_neuron_by_id(&self, id: usize) -> Option<Neuron> {
    let mut n = None;
    
    for neuron in &self.neurons {
      if neuron.id() == id {
        n = Some(neuron.clone());
        break;
      }
    }
    
    n
  }
  
  pub fn get_neurons_by_id(&self, ids: &Vec<usize>) -> Vec<Neuron> {
    let mut neuron_subset = Vec::new();
    for neuron in &self.neurons {
      if ids.contains(&neuron.id()) {
        neuron_subset.push(neuron.clone());
      }
    }
    
    neuron_subset
  }
  
  pub fn get_gene_by_innovation(&self, inno: usize) -> Option<Gene> {
    let mut gene = None;
    for i in 0..self.genes.len() {
      if inno == self.genes[i].innovation() {
        gene = Some(self.genes[i].clone());
        break;
      }
    }
    
    gene
  }
  
  
  pub fn has_link(&self, link: &Gene) -> bool {
    let mut has_link = false;
    for gene in &self.genes {
      if gene.from() == link.from() && gene.to() == link.to() {
        has_link = true;
        break;
      }
    }
    
    has_link
  }
  
  pub fn get_link(&self, link: &Gene) -> Gene {
    let mut return_gene = Gene::new();
    for gene in &self.genes {
      if gene.from() == link.from() && gene.to() == link.to() {
        return_gene = gene.copy();
        break;
      }
    }
    
    return_gene
  }
  
  pub fn has_neuron_between(&self, node_from: usize, node_to: usize) -> Option<(usize, usize)> {
    let mut neuron_linked = None;
    //println!("Node from: {} node to {}", node_from, node_to);
    for i in 0..self.genes.len() {
      if self.genes[i].from() == node_from && self.genes[i].to() != node_to {
        //println!("Prospect Gene1 [from: {}, to: {}]", self.genes[i].from(), self.genes[i].to());
        for j in 0..self.genes.len() {
          if i == j {
            continue;
          }
          
          if self.genes[j].to() == node_to && self.genes[j].from() != node_from {
            //println!("Prospect Gene2 [from: {}, to: {}]", self.genes[j].from(), self.genes[j].to());
            if self.genes[i].to() == self.genes[j].from() {
              neuron_linked = Some((self.genes[i].innovation(), self.genes[j].innovation()));
              break;
            }
            /*for k in 0..self.neurons.len() {
              if self.neurons[k].id() == self.genes[j].to() && 
                 self.genes[j].from() != node_from &&
                 !self.neurons[k].is_input() {
                 if self.neurons[k].id() == self.genes[i].from() && 
                    self.genes[i].to() != node_to {
                      neuron_linked = Some((self.genes[i].innovation(), self.genes[j].innovation(), self.neurons[k].id()));
                    }
                for l in 0..self.neurons.len() {
                  if k == l {
                    continue;
                  }
                  
                  if self.neurons[l].id() == self.genes[j].from() && self.neurons[l].id() != {
                    neuron_linked = Some((self.genes[i].innovation(), self.genes[j].innovation(), self.neurons[k].id()));
                  }
                }
              }
            }*/
          }
        }
      }
      if neuron_linked.is_some() {
        break;
      }
    }
    
    neuron_linked
  }
  
  pub fn set_gene_weight_by_innovation(&mut self, inno: usize, w: f64) {
    for i in 0..self.genes.len() {
      if inno == self.genes[i].innovation() {
        self.genes[i].set_weight(w);
        break;
      }
    }
  }
  /* Dont use is global not meant to be here
  pub fn random_neuron(&self, genes: &Vec<Gene>, non_input: bool, rng: &mut ThreadRng) -> Neuron {
    let mut idx = 0;
    let mut neuron_not_found = true;
    
    let mut attempts = 0;
    
    while neuron_not_found {
      let rand = rng.gen_range(0, self.neurons.len());
      
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
  }*/
  
  pub fn add_gene(&mut self, gene: &Gene) {
    self.genes.push(gene.copy());
  }
  
  pub fn add_neuron(&mut self, neuron: &Neuron) {
    self.neurons.push(neuron.clone());
  }
  
  pub fn next_id(&mut self) -> usize {
    self.last_id += 1;
    self.last_id
  }
  
  pub fn next_innovation(&mut self) -> usize {
    self.innovation += 1;
    self.innovation
  }
  
  pub fn set_innovation(&mut self, inno: usize) {
    self.innovation = inno;
  }
  
  pub fn set_inputs(&mut self, inputs: usize) {
    self.num_inputs = inputs;
  }
  
  pub fn set_outputs(&mut self, outputs: usize) {
    self.num_outputs = outputs;
  }
  
  pub fn set_last_id(&mut self, id: usize) {
    self.last_id = id;
  }
}





















