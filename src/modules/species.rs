use crate::modules::{Genome, NetworkInfo, NeatParams};
use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Species {
  num_inputs: usize,
  num_outputs: usize,
  top_fitness: f64,
  average_fitness: f64,
  staleness: u32,
  genomes: Vec<Genome>,
}

impl Species {
  pub fn new(num_inputs: usize, num_outputs: usize) -> Species {
    Species {
      num_inputs,
      num_outputs,
      top_fitness: 0.0,
      average_fitness: 0.0,
      staleness: 0,
      genomes: Vec::new(),
    }
  }
  
  pub fn average_fitness(&self) -> f64 {
    self.average_fitness
  }
  
  pub fn top_fitness(&self) -> f64 {
    self.top_fitness
  }
  
  pub fn staleness(&self) -> u32 {
    self.staleness
  }
  
  pub fn genomes(&self) -> &Vec<Genome> {
    &self.genomes
  }
  
  pub fn set_top_fitness(&mut self, f: f64) {
    self.top_fitness = f;
  }
  
  pub fn set_average_fitness(&mut self, f: f64) {
    self.average_fitness = f;
  }
  
  pub fn set_staleness(&mut self, s: u32) {
    self.staleness = s;
  }
  
  pub fn mut_genomes(&mut self) -> &mut Vec<Genome> {
    &mut self.genomes
  }
  
  pub fn sort_genomes(&mut self) {
    self.genomes.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
    self.genomes.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
  }
  
  pub fn calculate_average_fitness(&mut self) {
    let mut total = 0.0;
    for i in 0..self.genomes.len() {
      total += self.genomes[i].global_rank() as f64;
    }
    
    self.average_fitness = total / self.genomes.len() as f64;
  }
  
  pub fn breed_child(&self, params: &NeatParams, network_info: &mut NetworkInfo, rng: &mut ThreadRng) -> Genome {
    let mut child = Genome::new(self.num_inputs, self.num_outputs, params);
    
    if rng.gen::<f64>() < params.crossover_chance() {
      let g1 = &self.genomes[rng.gen_range(0..self.genomes.len())];
      let g2 = &self.genomes[rng.gen_range(0..self.genomes.len())];
      
      child = Genome::crossover(g1.copy(), g2.copy(), rng, params);
    } else {
      let g = &self.genomes[rng.gen_range(0..self.genomes.len())];
      child = g.copy();
    }
    
    child.mutate(network_info, rng);
    
    child
  }
}
