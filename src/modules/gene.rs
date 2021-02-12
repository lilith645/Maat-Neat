use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;
pub use bincode::{deserialize, serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gene {
  into: usize,
  out: usize,
  weight: f64,
  enabled: bool,
  innovation: usize,
}

impl Gene {
  pub fn new() -> Gene {
    Gene {
      into: 0,
      out: 0,
      weight: 0.0,
      enabled: true,
      innovation: 0,
    }
  }
  
  pub fn copy(&self) -> Gene {
    let mut gene = Gene::new();
    gene.set_from(self.into);
    gene.set_to(self.out);
    gene.set_weight(self.weight);
    gene.set_enabled(self.is_enabled());
    gene.set_innovation(self.innovation);
    
    gene
  }
  
  pub fn from(&self) -> usize {
    self.into
  }
  
  pub fn to(&self) -> usize {
    self.out
  }
  
  pub fn weight(&self) -> f64 {
    self.weight
  }
  
  pub fn is_enabled(&self) -> bool {
    self.enabled
  }
  
  pub fn innovation(&self) -> usize {
    self.innovation
  }
  
  pub fn set_from(&mut self, x: usize) {
    self.into = x;
  }
  
  pub fn set_to(&mut self, x: usize) {
    self.out = x;
  }
  
  pub fn set_weight(&mut self, x: f64) {
    self.weight = x;
  }
  
  pub fn set_enabled(&mut self, x: bool) {
    self.enabled = x;
  }
  
  pub fn set_innovation(&mut self, x: usize) {
    self.innovation = x;
  }
}
