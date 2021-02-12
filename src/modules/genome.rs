use crate::modules::{Gene, NetworkInfo, NeatParams, Network, Neuron, NodeType};
use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;
use std::ops::Deref;
pub use bincode::{deserialize, serialize};

use std::fs::File;
use std::io::Write;
use std::io::BufReader;
use std::io::Read;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Genome {
  num_inputs: usize,
  num_outputs: usize,
  fitness: f64,
  adjusted_fitness: f64,
  genes: Vec<Gene>,
  network: Option<Network>,
  max_neuron: usize,
  global_rank: u32,
  params: NeatParams,
}

impl Genome {
  pub fn new(num_inputs: usize, num_outputs: usize, params: &NeatParams) -> Genome {
    Genome {
      num_inputs,
      num_outputs,
      fitness: 0.0,
      adjusted_fitness: 0.0,
      genes: Vec::new(),
      network: None,
      max_neuron: 0,
      global_rank: 0,
      params: params.clone(),
    }
  }
  
  pub fn save(&self, location: &str) {
    let mut write = File::create(location).unwrap();
    write.write_all(&self.serialise());
  }
  
  pub fn load(&mut self, location: &str) {
    let mut file = File::open(location).unwrap();
    let mut buf_reader = BufReader::new(file);
    let mut contents = vec![];
    buf_reader.read_to_end(&mut contents).unwrap();
    self.genes.clear();
    self.network = None;
    if let Some(genome) = Genome::deserialise(&contents) {
      //println!("Genome: {:?}", genome);
      *self = genome;
      
      if let Some(network) = &mut self.network {
        network.reset_values();
      }
      
      //println!("Genome: {:?}", self);
     // println!("Genome: {} Ouputs: {} | Loaded!", location, self.num_outputs);
    }
  }
  
  pub fn serialise(&self) -> Vec<u8> {
    bincode::serialize(&self).unwrap()
  }
  
  pub fn deserialise(serialised: &[u8]) -> Option<Genome> {
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
  
  pub fn basic(num_inputs: usize, num_outputs: usize, params: &NeatParams, network_info: &mut NetworkInfo, rng: &mut ThreadRng) -> Genome {
    let mut g = Genome::new(num_inputs, num_outputs, params);
    
    g.set_max_neuron(num_inputs+num_outputs);
    
    g.mutate(network_info, rng);
    
    g
  }
  
  pub fn fully_connected(num_inputs: usize, num_outputs: usize, params: &NeatParams, network_info: &mut NetworkInfo, rng: &mut ThreadRng) -> Genome {
    let mut g = Genome::new(num_inputs, num_outputs, params);
    
    g.set_max_neuron(num_inputs+num_outputs);
    
    for from in 0..num_inputs {
      for to in 0..num_outputs {
        let mut new_link = Gene::new();
        new_link.set_from(from);
        new_link.set_to(to+params.max_nodes());
        if network_info.has_link(&new_link) {
          new_link = network_info.get_link(&new_link);
          new_link.set_weight(rng.gen::<f64>()*4.0-2.0);
        } else {
          new_link.set_innovation(network_info.next_innovation());
          new_link.set_weight(rng.gen::<f64>()*4.0-2.0);
          network_info.add_gene(&new_link);
        }
        
        g.add_gene(new_link);
      }
    }
    
    g.mutate(network_info, rng);
    
    g
  }
  
  pub fn network(&self) -> Network {
    self.network.clone().unwrap()
  }
  
  pub fn num_inputs(&self) -> usize {
    self.num_inputs
  }
  
  pub fn num_outputs(&self) -> usize {
    self.num_outputs
  }
  
  pub fn fitness(&self) -> f64 {
    self.fitness
  }
  
  pub fn global_rank(&self) -> u32 {
    self.global_rank
  }
  
  pub fn max_neuron(&self) -> usize {
    self.max_neuron
  }
  
  pub fn genes(&self) -> &Vec<Gene> {
    &self.genes
  }
  
  pub fn copy_genes(&self) -> Vec<Gene> {
    self.genes.clone()
  }
  
  pub fn copy(&self) -> Genome {
    let mut g = Genome::new(self.num_inputs, self.num_outputs, &self.params);
    g.set_genes(self.copy_genes());
    g.set_max_neuron(self.max_neuron);
    g.set_params(self.params.clone());
    
    g
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
  
  pub fn params(&self) -> NeatParams {
    self.params.clone()
  }
  
  pub fn set_fitness(&mut self, f: f64) {
    self.fitness = f;
  }
  
  pub fn add_gene(&mut self, gene: Gene) {
    self.genes.push(gene);
  }
  
  pub fn set_genes(&mut self, genes: Vec<Gene>) {
    self.genes = genes;
  }
  
  pub fn set_params(&mut self, params: NeatParams) {
    self.params = params;
  }
  
  pub fn set_max_neuron(&mut self, neuron: usize) {
    self.max_neuron = neuron;
  }
  
  pub fn set_global_rank(&mut self, rank: u32) {
    self.global_rank = rank;
  }
  
  pub fn network_neurons(&self) -> Vec<Neuron> {
    let mut neurons = Vec::new();
    if let Some(network) = &self.network {
      neurons.append(&mut network.neurons().clone());
    }
    neurons
  }
  
  pub fn generate_network(&mut self, network_info: &NetworkInfo) {
    self.genes.sort_by(|a, b| a.to().partial_cmp(&b.to()).unwrap());
    
    self.network = Some(Network::new(self.num_inputs, self.num_outputs, &self.params, network_info, &self.genes));
  }
  
  pub fn evaluate_network(&mut self, inputs: Vec<f64>) -> Vec<f64> {
    if self.num_inputs != inputs.len() {
      panic!("Incorrect number of inputs (Expected: {}, Actual: {})", self.num_inputs, inputs.len());
    }
    
    let mut output = Vec::new();
    
    //self.generate_network(network_info);
    
    if let Some(network) = &mut self.network {
      network.reset_values();
      network.add_inputs(inputs);
      output = network.evaluate();
    }
    
    output
  }
  
  pub fn crossover(mut g1: Genome, mut g2: Genome, rng: &mut ThreadRng, params: &NeatParams) -> Genome {
    if g2.fitness() > g1.fitness() {
      let temp = g1;
      g1 = g2;
      g2 = temp;
    }
    
    let mut child = Genome::new(g1.num_inputs(), g1.num_outputs(), params);
    
    let mut g2_innovations = Vec::new();
    for gene in g2.copy_genes() {
      g2_innovations.push((gene.innovation(), gene));
    }
    
    for i in 0..g1.genes().len() {
      let g1_g = &g1.copy_genes()[i];
      
      let mut g2_has_g1_gene = false;
      let mut g2_enabled = false;
      let mut g2 = Gene::new();
      for i in 0..g2_innovations.len() {
        if g2_innovations[i].0 == g1_g.innovation() {
          g2_has_g1_gene = true;
          g2 = g2_innovations[i].1.copy();
          g2_enabled = g2.is_enabled();
          break;
        }
      }
      
      if g2_has_g1_gene && (rng.gen::<f32>() > 0.5) && g2_enabled {
        child.add_gene(g2.copy());
      } else {
        child.add_gene(g1_g.copy());
      }
    }
    
    child.set_max_neuron((g1.max_neuron()).max(g2.max_neuron()));
    
    // copy mutation rate???
    child.set_params(g1.params());
    
    child
  }
  
  pub fn disjoint(genes1: &Vec<Gene>, genes2: &Vec<Gene>) -> f64 {
    let mut disjoint_genes = 0.0;
    let mut inno1 = Vec::new();
    let mut inno2 = Vec::new();
    
    for gene in genes1 {
      inno1.push(gene.innovation());
    }
    for gene in genes2 {
      inno2.push(gene.innovation());
    }
    
    for inno in &inno1 {
      if !inno2.contains(inno) {
        disjoint_genes += 1.0;
      }
    }
    
    for inno in &inno2 {
      if !inno1.contains(inno) {
        disjoint_genes += 1.0;
      }
    }
    
    let n = genes1.len().max(genes2.len()).max(1);
    
    disjoint_genes / n as f64
  }
  
  pub fn weights(genes1: &Vec<Gene>, genes2: &Vec<Gene>) -> f64 {
    let mut sum = 0.0;
    let mut coincident = 0;
    let mut inno2 = Vec::new();
    
    for i in 0..genes2.len() {
      inno2.push(genes2[i].innovation());
    }
    
    for i in 0..genes1.len() {
      if inno2.contains(&genes1[i].innovation()) {
        let gene2_weight = {
          let mut gene2_weight = 0.0;
          for j in 0..inno2.len() {
            if inno2[j] == genes1[i].innovation() {
              gene2_weight = genes2[j].weight();
            }
          }
          
          gene2_weight
        };
        
        sum += (genes1[i].weight() - gene2_weight).abs();
        coincident += 1;
      }
    }
    
    sum / (coincident).max(1) as f64
  }
  
  pub fn same_species(&self, genome: &Genome, params: &NeatParams) -> bool {
    let dd = self.params.delta_disjoint() * Genome::disjoint(&self.genes(), genome.genes());
    let dw = self.params.delta_weights() * Genome::weights(&self.genes(), genome.genes());
    
    dd + dw < params.delta_threshold()
  }
  
  pub fn mutate(&mut self, network_info: &mut NetworkInfo, rng: &mut ThreadRng) {
    if rng.gen_range(0..2) == 0 { // 0 or 1
      self.params.multiply_mutation_rates(0.95);
    } else {
      self.params.multiply_mutation_rates(1.05263);
    }
    
    if rng.gen::<f64>() < self.params.mutate_connection_chance() {
      self.point_mutate(rng);
    }
    
    let mut p = self.params.link_mutation_chance();
    while p > 0.0 {
      if rng.gen::<f64>() < p {
        self.link_mutate(false, network_info, rng);
      }
      p -= 1.0;
    }
    
    p = self.params.bias_mutation_chance();
    while p > 0.0 {
      if rng.gen::<f64>() < p {
        self.link_mutate(true, network_info, rng);
      }
      p -= 1.0;
    }
    
    p = self.params.node_mutation_chance();
    while p > 0.0 {
      if rng.gen::<f64>() < p {
        self.node_mutate(network_info, rng);
      }
      p -= 1.0;
    }
    
    p = self.params.enable_mutation_chance();
    while p > 0.0 {
      if rng.gen::<f64>() < p {
        self.enable_disable_mutate(true, rng);
      }
      p -= 1.0;
    }
    
    p = self.params.disable_mutation_chance();
    while p > 0.0 {
      if rng.gen::<f64>() < p {
        self.enable_disable_mutate(false, rng);
      }
      p -= 1.0;
    }
  }
  
  pub fn point_mutate(&mut self, rng: &mut ThreadRng) {
    for i in 0..self.genes.len() {
      let gene = &mut self.genes[i];
      if rng.gen::<f64>() < self.params.perturb_chance() {
        gene.set_weight(gene.weight() + rng.gen::<f64>() * self.params.step_size()*2.0 - self.params.step_size());
      } else {
        gene.set_weight(rng.gen::<f64>()*4.0-2.0);
      }
    }
  }
  
  pub fn link_mutate(&mut self, force_bias: bool, network_info: &mut NetworkInfo, rng: &mut ThreadRng) {
    
    if self.network.is_none() {
      self.generate_network(network_info);
    }
    
    let network = self.network.as_ref().unwrap();
    
    let mut neuron1 = network.random_neuron(false, rng);
    let mut neuron2 = network.random_neuron(true, rng);
    
    if neuron1.is_input() && neuron2.is_input() {
      return;
    }
    
    let mut new_link = Gene::new();
    if neuron2.is_input() {
      let temp = neuron1;
      neuron1 = neuron2;
      neuron2 = temp;
    }
    
    new_link.set_from(neuron1.id());
    new_link.set_to(neuron2.id());
    
    if force_bias {
      new_link.set_from(self.num_inputs-1);
    }
    
    if self.has_link(&new_link)  {
      /*for gene in &mut self.genes {
        if gene.from() == new_link.from() && gene.to() == new_link.to() {
          gene.set_weight(rng.gen::<f64>()*4.0-2.0);
          gene.set_enabled(true);
          break;
        }
      }*/
      return;
    }
    
    if network_info.has_link(&new_link) {
      new_link = network_info.get_link(&new_link);
      new_link.set_weight(rng.gen::<f64>()*4.0-2.0);
    } else {
      new_link.set_innovation(network_info.next_innovation());
      new_link.set_weight(rng.gen::<f64>()*4.0-2.0);
      network_info.add_gene(&new_link);
    }
    
    self.genes.push(new_link);
  }
  
  pub fn node_mutate(&mut self, network_info: &mut NetworkInfo, rng: &mut ThreadRng) {
    if self.genes.len() == 0 {
      return;
    }
    
    self.max_neuron += 1;
    
    let g_len = self.genes.len();
    let gene = &mut self.genes[rng.gen_range(0..g_len)];
    if !gene.is_enabled() {
      return;
    }
    
    let to = gene.to();
    let from = gene.from();
    
    let x = (network_info.get_neuron_by_id(gene.from()).unwrap().x() + network_info.get_neuron_by_id(gene.to()).unwrap().x()) * 0.5;
    let y = (network_info.get_neuron_by_id(gene.from()).unwrap().y() + network_info.get_neuron_by_id(gene.to()).unwrap().y()) * 0.5 + rng.gen::<f32>() * 0.1 - 0.05;
    
    let mut gene1;
    let mut gene2;
    let mut neuron;
    
    let mut already_in_network = false;
    
    neuron = Neuron::new(NodeType::Hidden, x, y, network_info.next_id());
    
    if let Some((inno_from, inno_to)) = network_info.has_neuron_between(from, to) {
      gene1 = network_info.get_gene_by_innovation(inno_from).unwrap();
      gene2 = network_info.get_gene_by_innovation(inno_to).unwrap();
      already_in_network = true;
    } else {
      gene1 = gene.copy();
      gene1.set_to(neuron.id());
      gene1.set_innovation(network_info.next_innovation());
      
      gene2 = gene.copy();
      gene2.set_from(neuron.id());
      gene2.set_innovation(network_info.next_innovation());
    }
    
    gene.set_enabled(false);
    
    gene1.set_weight(1.0);
    gene1.set_enabled(true);
    
    gene2.set_enabled(true);
    
    self.genes.push(gene1.copy());
    self.genes.push(gene2.copy());
    
    if !already_in_network {
      network_info.add_neuron(&neuron);
      network_info.add_gene(&gene1);
      network_info.add_gene(&gene2);
    }
  }

  pub fn enable_disable_mutate(&mut self, enable: bool, rng: &mut ThreadRng) {
    let mut candidates = Vec::new();
    
    for i in 0..self.genes.len() {
      if self.genes[i].is_enabled() == !enable {
        candidates.push(i);
      }
    }
    
    if candidates.len() == 0 {
      return;
    }
    
    let mut gene = &mut self.genes[candidates[rng.gen_range(0..candidates.len())]];
    let gene_is_enabled = gene.is_enabled();
    gene.set_enabled(!gene_is_enabled);
  }
  
  pub fn reposition_neurons(&mut self, rng: &mut ThreadRng) {
    if let Some(network) = &mut self.network {
      network.reposition_neurons(rng);
    }
  }
}




















