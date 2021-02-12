use crate::modules::{NetworkInfo, NeatParams, Species, Genome};
use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;

use std::time::{Duration, Instant};

use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use std::io::Read;

#[derive(Clone, Serialize, Deserialize)]
pub struct Pool {
  num_inputs: usize,
  num_outputs: usize,
  generation: usize,
  species: Vec<Species>,
  innovation: u32,
  current_species: usize,
  current_genome: usize,
  max_fitness: f64,
  population: usize,
  //last_gen: Instant,
  top: Option<Genome>,
}

impl Pool {
  pub fn new(num_inputs: usize, num_outputs: usize, population: usize) -> Pool {
    Pool {
      num_inputs: num_inputs+1, // bias
      num_outputs,
      generation: 0,
      species: Vec::new(),
      innovation: 0,
      current_species: 0,
      current_genome: 0,
      max_fitness: 0.0,
      population,
      //last_gen: Instant::now(),
      top: None,
    }
  }
  
  pub fn serialise(&self) -> Vec<u8> {
    bincode::serialize(&self).unwrap()
  }
  
  pub fn deserialise(serialised: &[u8]) -> Option<Pool> {
    match bincode::deserialize(&serialised) {
      Ok(data) => {
        Some(data)
      },
      Err(e) => {
        println!("{:?}", e);
        None
      }
    }
  }
  
  pub fn save(&self, location: &str) {
    let mut write = File::create(location).unwrap();
    write.write_all(&self.serialise());
  }
  
  pub fn load(&mut self, network_info: &mut NetworkInfo, params: &NeatParams, location: &str) {
    let mut file = File::open(location).unwrap();
    let mut buf_reader = BufReader::new(file);
    let mut contents = vec![];
    buf_reader.read_to_end(&mut contents).unwrap();
    
    if let Some(pool) = Pool::deserialise(&contents) {
      let s = pool.species().clone();
      *self = pool;
      self.species = s;
    }
    
    *network_info = NetworkInfo::new_empty();
    
    
    let mut highest_inno = 0;
    let mut last_id = 0;
    
    for species in &self.species {
      for source_genome in species.genomes() {
        for i in 0..source_genome.genes().len() {
          network_info.set_inputs(source_genome.num_inputs());
          network_info.set_outputs(source_genome.num_outputs());
          
          let gene = source_genome.genes()[i].copy();
          
          if gene.innovation() > highest_inno {
            highest_inno = gene.innovation();
          }
          
          let neurons = source_genome.network_neurons();
          for neuron in neurons {
            network_info.add_neuron(&neuron);
            if neuron.id() < params.max_nodes() {
              if neuron.id() > last_id {
                last_id = neuron.id();
              }
            }
          }
          
          if !network_info.has_link(&gene) {
            network_info.add_gene(&gene);
          }
        }
      }
    }
    
    network_info.set_innovation(highest_inno);
    network_info.set_last_id(last_id);
    
    //self.last_gen = Instant::now();
  }
  
  pub fn generation(&self) -> usize {
    self.generation
  }
  
  pub fn initialise_pool(&mut self, params: &NeatParams, network_info: &mut NetworkInfo, rng: &mut ThreadRng) {
    for i in 0..self.population {
      let basic = Genome::fully_connected(self.num_inputs, self.num_outputs, params, network_info, rng);
      self.add_to_species(&basic, params);
    }
    
    for i in 0..self.species.len() {
      for j in 0..self.species[i].genomes().len() {
        self.species[i].mut_genomes()[j].generate_network(&network_info);
      }
    }
    
    //self.last_gen = Instant::now();
  }
  
  pub fn initialise_pool_from_genome(&mut self, params: &NeatParams, network_info: &mut NetworkInfo, rng: &mut ThreadRng, location: &str) {
    let mut source_genome = Genome::new(self.num_inputs, self.num_outputs, params);
    source_genome.load(location);
    
    *network_info = NetworkInfo::new_empty();
    
    network_info.set_inputs(source_genome.num_inputs());
    network_info.set_outputs(source_genome.num_outputs());
    
    
    let mut highest_inno = 0;
    for i in 0..source_genome.genes().len() {
      let gene = source_genome.genes()[i].copy();
      
      if gene.innovation() > highest_inno {
        highest_inno = gene.innovation();
      }
      
      network_info.add_gene(&gene);
    }
    network_info.set_innovation(highest_inno);
    
    let mut last_id = 0;
    
    let neurons = source_genome.network_neurons();
    for neuron in neurons {
      network_info.add_neuron(&neuron);
      if neuron.id() < params.max_nodes() {
        last_id = neuron.id();
      }
    }
    
    network_info.set_last_id(last_id);
    
    //source_genome.set_params(params.clone());
    self.add_to_species(&source_genome, params);
    
    for i in 1..self.population {
      let mut basic = source_genome.clone();
      basic.mutate(network_info, rng);
      self.add_to_species(&basic, params);
    }
    
    for i in 0..self.species.len() {
      for j in 0..self.species[i].genomes().len() {
        self.species[i].mut_genomes()[j].generate_network(&network_info);
      }
    }
    
    //self.last_gen = Instant::now();
  }
  
  pub fn new_innovation(&mut self) -> u32 {
    self.innovation += 1;
    self.innovation
  }
  
  pub fn species(&self) -> &Vec<Species> {
    &self.species
  }
  
  pub fn mut_species(&mut self) -> &mut Vec<Species> {
    &mut self.species
  }
  
  pub fn mut_current_genome(&mut self) -> &mut Genome {
    &mut self.species[self.current_species].mut_genomes()[self.current_genome]
  }
  
  pub fn current_genome(&self) -> &Genome {
    &self.species[self.current_species].genomes()[self.current_genome]
  }
  
  pub fn generate_current(&mut self, params: &NeatParams, network_info: &NetworkInfo) {
    let species = &mut self.species[self.current_species];
    let mut genome = &mut species.mut_genomes()[self.current_genome];
    
    genome.generate_network(network_info);
  }
  
  pub fn evaluate_current(&mut self, mut inputs: Vec<f64>, params: &NeatParams, network_info: &NetworkInfo) -> Vec<f64> {
    let species = &mut self.species[self.current_species];
    let mut genome = &mut species.mut_genomes()[self.current_genome];
    
    let mut inputs = inputs;
    inputs.insert(0, 1.0); // Bias
    
    //println!("ec Inputs: {:?}", inputs);
    genome.evaluate_network(inputs)
  }
  
  pub fn total_average_fitness(&mut self) -> f64 {
    let mut total = 0.0;
    for i in 0..self.species.len() {
      self.species[i].calculate_average_fitness();
      total += self.species[i].average_fitness();
    }
    
    total
  }
  
  pub fn set_fitness_current(&mut self, fitness: f64) {
    self.species[self.current_species].mut_genomes()[self.current_genome].set_fitness(fitness);
    if fitness > self.max_fitness {
      self.max_fitness = fitness;
      self.top = Some(self.species[self.current_species].mut_genomes()[self.current_genome].clone());
    }
  }
  
  pub fn next_genome(&mut self, network_info: &mut NetworkInfo, params: &mut NeatParams, rng: &mut ThreadRng, last_gen: &mut Instant) {
    self.current_genome += 1;
    if self.current_genome >= self.species[self.current_species].genomes().len() {
      self.current_genome = 0;
      self.current_species += 1;
      if self.current_species >= self.species.len() {
        self.new_generation(network_info, params, rng, last_gen);
        self.current_species = 0;
      }
    }
  }
  
  pub fn new_generation(&mut self, network_info: &mut NetworkInfo, params: &mut NeatParams, rng: &mut ThreadRng, last_gen: &mut Instant) {
    let mut num_species = self.species.len();
    let mut highest_fitness = -999999999.0;
    let mut average_fitness = 0.0;
    let mut count = 0;
    for i in 0..self.species.len() {
      for j in 0..self.species[i].genomes().len() {
        count += 1;
        let g_fitness = self.species[i].genomes()[j].fitness();
        average_fitness += g_fitness;
        if g_fitness > highest_fitness {
          highest_fitness = g_fitness;
        }
      }
    }
    average_fitness /= count as f64;
    
    let count = self.species.len();
    if count != params.target_species() {
      if count > params.target_species() {
        params.increase_delta_threshold();
      } else {
        params.decrease_delta_threshold();
      }
    }
    
    self.cull_species(false); // cull bottom half of each species
    if self.species.len() == 0 {
      panic!("At least one species should exist at this point (Cull species (false))");
    }
    self.rank_globally();
    self.remove_stale_species(params);
    if self.species.len() == 0 {
      panic!("At least one species should exist at this point (Remove Stale)");
    }
    self.rank_globally();
    for i in 0..self.species.len() {
      self.species[i].calculate_average_fitness();
    }
    self.remove_weak_species(params);
    if self.species.len() == 0 {
      panic!("At least one species should exist at this point (Remove Weak)");
    }
    let sum = self.total_average_fitness();
    let mut children = Vec::new();
    for i in 0..self.species.len() {
      let breed = (self.species[i].average_fitness() / sum * self.population as f64).floor() as u32 - 1;
      for _ in 0..breed {
        children.push(self.species[i].breed_child(params, network_info, rng));
      }
    }
    self.cull_species(true); // cull all but top member of each species
    if self.species.len() == 0 {
      panic!("At least one species should exist at this point (Cull species (true))");
    }
    while children.len() + self.species.len() < self.population {
      let i = if self.species.len() > 1 { rng.gen_range(0..self.species.len()) } else { 0 };
      children.push(self.species[i].breed_child(params, network_info, rng));
    }
    
    for i in 0..children.len() {
      self.add_to_species(&children[i], &params);
    }
    
    self.generation += 1;
    
    for i in 0..self.species.len() {
      for j in 0..self.species[i].genomes().len() {
        self.species[i].mut_genomes()[j].set_fitness(0.0);
        self.species[i].mut_genomes()[j].generate_network(network_info);
      }
    }
    
    println!("Generation: {} ({} seconds)", self.generation-1, last_gen.elapsed().as_secs());
    println!("    Threshold      : {}", params.delta_threshold());
    println!("    Species        : {}", self.species.len());
    println!("    Top Fitness    : {}", highest_fitness);
    println!("    Average Fitness: {}", average_fitness);
    
    *last_gen = Instant::now()
  }
  
  pub fn cull_species(&mut self, cut_to_one: bool) {
    for i in 0..self.species.len() {
      self.species[i].sort_genomes();
      
      let mut remaining = (self.species[i].genomes().len() as f32*0.5).ceil() as usize;
      if cut_to_one {
        remaining = 1;
      }
      
      while self.species[i].genomes().len() > remaining {
        self.species[i].mut_genomes().remove(0);
      }
    }
  }
  
  pub fn remove_stale_species(&mut self, params: &NeatParams) {
    let mut survived = Vec::new();
    
    for i in 0..self.species.len() {
      self.species[i].sort_genomes();
      
      let idx = self.species[i].genomes().len()-1;
      let genome_fitness = self.species[i].genomes()[idx].fitness();
      
      if genome_fitness > self.species[i].top_fitness() {
        self.species[i].set_top_fitness(genome_fitness);
        self.species[i].set_staleness(0);
      } else {
        let stale = self.species[i].staleness();
        self.species[i].set_staleness(stale + 1);
      }
      
      if self.species[i].staleness() < params.stale_species() || 
         self.species[i].top_fitness() >= self.max_fitness {
        survived.push(self.species[i].clone());
      }
    }
    
    self.species = survived;
  }
  
  pub fn remove_weak_species(&mut self, params: &NeatParams) {
    let mut survived = Vec::new();
    
    let mut sum = self.total_average_fitness();
    for i in 0..self.species.len() {
     // println!("Total Avg: {}, Species Avg: {}", sum, self.species[i].average_fitness());
      let breed = ((self.species[i].average_fitness() / sum) * self.population as f64).floor();
      if breed >= 1.0 {
        survived.push(self.species[i].clone());
      }
    }
    
    self.species = survived;
  }
  
  pub fn add_to_species(&mut self, child: &Genome, params: &NeatParams) {
    let mut found_species = false;
    
    for i in 0..self.species.len() {
      if !found_species && child.same_species(&self.species[i].genomes()[0], params) {
        self.species[i].mut_genomes().push(child.copy());
        found_species = true;
        break;
      }
    }
    
    if !found_species {
      let mut new_species = Species::new(self.num_inputs, self.num_outputs);
      new_species.mut_genomes().push(child.copy());
      self.species.push(new_species);
    }
  }
  
  pub fn rank_globally(&mut self) {
    let mut global_ranks: Vec<((usize, usize), f64)> = Vec::new();
    
    for i in 0..self.species.len() {
      for j in 0..self.species[i].genomes().len() {
        global_ranks.push(((i, j), self.species[i].genomes()[j].fitness()));
      }
    }
    
    global_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    //println!("idx 0 {} idx last {}", global_ranks[0].1, global_ranks[global_ranks.len()-1].1);
    for i in 0..global_ranks.len() {
      let s = (global_ranks[i].0).0;
      let g = (global_ranks[i].0).1;
      let f = global_ranks[i].1;
      self.species[s].mut_genomes()[g].set_global_rank(i as u32+1);
    }
    /*
    for i in 0..self.species.len() {
      for j in 0..self.species[i].genomes().len() {
        println!("Rank: {} Fit: {}", self.species[i].genomes()[j].global_rank(), self.species[i].genomes()[j].fitness());
      }
    }*/
  }
}














