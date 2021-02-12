#[macro_use]
pub extern crate serde_derive;
pub extern crate bincode;

mod modules;

use rand;
use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;

use std::time::Instant;

use crate::modules::{NeatParams, NetworkInfo, Pool, Genome, Neuron, Gene};

use crate::modules::THelper;

fn repeat_run_test(population: usize, w: f64, h: f64, turtle: &mut THelper, pool: &mut Pool, 
                   network_info: &mut NetworkInfo, params: &mut NeatParams, rng: &mut ThreadRng) {
  
  let inputs = [vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
  let expected_outputs = [vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
  
  let mut time = Instant::now();
  
  for i in 0..population*10 {
    let mut fitness = 0.0;
    
    for j in 0..inputs.len() {
      let mut output = pool.evaluate_current(inputs[j].clone(), &params, &network_info);
      if output[0] < 0.0 {
        output[0] = 0.0;
      }
      if output[0] > 0.0 {
        output[0] = 1.0;
      }
      
      fitness += pool.current_genome().genes().len() as f64;
    }
    
    pool.set_fitness_current(fitness);
    pool.next_genome(network_info, &params, rng, &mut time);
    pool.generate_current(&params, &network_info);
  }
  
  draw_genome(turtle, &pool.current_genome(), w*0.25, 0.0, w*0.5, h);
  
  for i in 0..2 {
    for j in 0..inputs.len() {
      let mut output = pool.evaluate_current(inputs[j].clone(), &params, &network_info);
      /*if output[0] < 0.0 {
        output[0] = 0.0;
      }
      if output[0] > 0.0 {
        output[0] = 1.0;
      }*/
      println!("Output: {:?}", output);
    }
  }
  /*
  for i in 0..pool.current_genome().network_neurons().len() {
    println!("{:?}", pool.current_genome().network_neurons()[i]);
  }*/
  
  pool.current_genome().save("./networks/test_genome.nn");
  
  *pool = Pool::new(2, 3, population);
  pool.initialise_pool_from_genome(&params, network_info, rng,
                                   "./networks/test_genome.nn");
  
  for i in 0..2 {
    for j in 0..inputs.len() {
      let mut output = pool.evaluate_current(inputs[j].clone(), &params, &network_info);
      println!("Output: {:?}", output);
    }
  }
  /*
  for i in 0..pool.current_genome().network_neurons().len() {
    println!("{:?}", pool.current_genome().network_neurons()[i]);
  }*/
  
  //println!("{:?}", pool.current_genome());
  
  draw_genome(turtle, &pool.current_genome(), -w*0.5, 0.0, w*0.5, h);
}

fn XOR_test(population: usize, w: f64, h: f64, turtle: &mut THelper, pool: &mut Pool, 
            network_info: &mut NetworkInfo, params: &mut NeatParams, rng: &mut ThreadRng) {
  
  let inputs = [vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
  let expected_outputs = [vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
  
  let mut time = Instant::now();
  
  loop {
    let mut best_fitness = -999999999999999.9;
    let mut best_outputs = Vec::new();
    let mut best_genome = None;
    for i in 0..population as usize {
      pool.generate_current(&params, &network_info);
      
      let mut genome_fitness = 4.9;
      let mut g_outputs = Vec::new();
      
      for j in 0..inputs.len() {
        let mut output = pool.evaluate_current(inputs[j].clone(), &params, &network_info);
        if output[0] < 0.0 {
          output[0] = 0.0;
        }
        if output[0] > 0.0 {
          output[0] = 1.0;
        }
        
        g_outputs.push(output[0].clone());
        let diff = (expected_outputs[j][0] as f32 - output[0] as f32);
        
        genome_fitness -= (diff * diff) as f64; 
      }
      //genome_fitness -= pool.current_genome().genes().len() as f64*0.001;
      pool.set_fitness_current(genome_fitness);
      if genome_fitness > best_fitness {
        best_fitness = genome_fitness;
        best_outputs = g_outputs.clone();
        best_genome = Some(pool.current_genome().clone());
      }
      
      if genome_fitness >= 4.9 {
        println!("Winning Outputs: {:?}", g_outputs.clone());
        turtle.clear();
        pool.mut_current_genome().reposition_neurons(rng);
        let g = pool.current_genome();
        draw_genome(turtle, &g, 0.0, 0.0, w, h);
       /* draw_genome_genes(turtle, g.genes(), &network_info, 0.0, 0.0, w, h);
        for n in &g.network_neurons() {
          draw_neuron(turtle, n, 0.0, 0.0, w, h);
        }*/
        
        turtle.wait();
      }
      
      pool.next_genome(network_info, &params, rng, &mut time);
    }
    println!("    Best Fitness: {}", best_fitness);
    println!("    Output: {:?}", best_outputs);
    println!("    Innovations: {}", network_info.innovations());
    if pool.generation() % 5 == 0 || pool.generation() == 2 {
      turtle.clear();
      let mut segments = ((pool.species().len() as f64).sqrt().ceil()) as usize;
      let mut segment_size_w = w as f64 / segments as f64;
      let mut segment_size_h = h as f64 / segments as f64;
      println!("Species: {}", pool.species().len());
      for i in 0..pool.species().len() {
        let x = (i%segments) as f64*segment_size_w + segment_size_w*0.5 - w*0.5;
        let y =  (i/segments) as f64*-segment_size_h - segment_size_h*0.5 + h*0.5;
        let w = segment_size_w;
        let h = segment_size_h;
        pool.mut_species()[i].mut_genomes()[0].reposition_neurons(rng);
        let genome = pool.species()[i].genomes()[0].clone();
        draw_genome(turtle, &genome, x, y, w, h);
       /* draw_genome_genes(turtle, genome.genes(), &network_info, x, y, w, h);
        for n in &genome.network_neurons() {
          draw_neuron(turtle, n, x, y, w, h);
        }*/
      }
    }
  }
}

fn test_load_save_genome(population: usize, w: f64, h: f64, turtle: &mut THelper, pool: &mut Pool, 
            network_info: &mut NetworkInfo, params: &mut NeatParams, rng: &mut ThreadRng) {
   /* 
  let mut genome = Genome::basic(3, 1, &params, network_info, rng);
  for i in 0..10 {
    genome.node_mutate(network_info, rng);
  }
  
  draw_genome(turtle, &genome, w*-0.25, 0.0, w*0.5, h);
  genome.save("./networks/test_genome.nn");
  */
  
  let mut new_genome = Genome::new(3, 3, &NeatParams::new());
  
  //new_genome.load("./networks/Gen393_Fit39.59_CF_1187.nn");
  
  //*pool = Pool::new(new_genome.num_inputs(), new_genome.num_outputs(), population);
  //pool.initialise_pool_from_genome(&params, network_info, rng, "./networks/Gen393_Fit39.59_CF_1187.nn");
  
  let mut time = Instant::now();
  
  for i in 0..100 {
    new_genome.mutate(network_info, rng);
    new_genome.generate_network(network_info);
    new_genome.evaluate_network(vec![1.0, 0.5, 0.5]);
  }
  draw_genome(turtle, &new_genome, w*0.25, 0.0, w*0.5, h);
  
  new_genome.save("./networks/test_genome.nn");
  
  *pool = Pool::new(new_genome.num_inputs(), new_genome.num_outputs(), population);
  pool.initialise_pool_from_genome(&params, network_info, rng, "./networks/test_genome.nn");
  
  let mut genome_reload = Genome::new(0, 0, &NeatParams::new());
  
  genome_reload.load("./networks/test_genome.nn");
  
  draw_genome(turtle, &genome_reload, -w*0.25, 0.0, w*0.5, h);
  
  if new_genome.genes().len() != genome_reload.genes().len() {
    panic!("Genes are not the same length (Expected: {}, Actual: {})", new_genome.genes().len(),
                                                                       genome_reload.genes().len());
  }
  
  if new_genome.network_neurons().len() != genome_reload.network_neurons().len() {
    panic!("Neuron count is different (Expected: {}, Actual: {})", new_genome.network_neurons().len(),
                                                                   genome_reload.network_neurons().len());
  }
  
  if new_genome.num_inputs() != genome_reload.num_inputs() {
    panic!("genome num_inputs do not match (Expected: {}, Actual: {})", 
           new_genome.num_inputs(), 
           genome_reload.num_inputs());
  }
  
  if new_genome.num_outputs() != genome_reload.num_outputs() {
    panic!("genome num_outputs do not match (Expected: {}, Actual: {})", 
           new_genome.num_outputs(), 
           genome_reload.num_outputs());
  }
  
  if new_genome.fitness() != genome_reload.fitness() {
    panic!("genome fitness do not match (Expected: {}, Actual: {})", 
           new_genome.fitness(), 
           genome_reload.fitness());
  }
  
  if new_genome.global_rank() != genome_reload.global_rank() {
    panic!("genome global_rank do not match (Expected: {}, Actual: {})", 
           new_genome.global_rank(), 
           genome_reload.global_rank());
  }
  
  if new_genome.max_neuron() != genome_reload.max_neuron() {
    panic!("genome max_neuron do not match (Expected: {}, Actual: {})", 
           new_genome.max_neuron(), 
           genome_reload.max_neuron());
  }
  
  for i in 0..new_genome.genes().len() {
    let gene1 = &new_genome.genes()[i];
    let gene2 = &new_genome.genes()[i];
    
    if gene1.from() != gene2.from() {
      panic!("gene from does not match (Expected: {}, Actual: {})", 
             gene1.from(), 
             gene2.from());
    }
    
    if gene1.to() != gene2.to() {
      panic!("gene to does not match (Expected: {}, Actual: {})", 
             gene1.to(), 
             gene2.to());
    }
    
    if gene1.weight() != gene2.weight() {
      panic!("gene weight does not match (Expected: {}, Actual: {})", 
             gene1.weight(), 
             gene2.weight());
    }
    
    if gene1.is_enabled() != gene2.is_enabled() {
      panic!("genes enabled mismatch (Expected: {}, Actual: {})", 
             gene1.is_enabled(), 
             gene2.is_enabled());
    }
    
    if gene1.innovation() != gene2.innovation() {
      panic!("genes innovation does not match (Expected: {}, Actual: {})", 
             gene1.innovation(), 
             gene2.innovation());
    }
  }
  
  
}

fn draw_genome_genes(turtle: &mut THelper, genes: &Vec<Gene>, network_info: &NetworkInfo, 
                     x_offset: f64, y_offset: f64, w: f64, h: f64) {
  for con in genes {
    let from = con.from();
    let to = con.to();
    
    if let Some(n_from) = network_info.get_neuron_by_id(from) {
      if let Some(n_to) = network_info.get_neuron_by_id(to) {
        let x = n_from.x() as f64 * w - w*0.5 + x_offset;
        let y = n_from.y() as f64 * h - h*0.5 + y_offset;
        let x1 = n_to.x() as f64 * w - w*0.5 + x_offset;
        let y1 = n_to.y() as f64 * h - h*0.5 + y_offset;
        
        if con.is_enabled() {
          turtle.set_colour("green");
        } else {
          turtle.set_colour("red");
        }
        
        let weight = con.weight();
        
        let pen_size = ((weight + 2.0) * 2.0).max(0.001);
        turtle.pen_size(pen_size);
        
        turtle.draw_line(x, y, x1, y1);
        turtle.pen_size(1.0);
      }
    }
  }
}

fn draw_neuron(turtle: &mut THelper, neuron: &Neuron,
               x_offset: f64, y_offset: f64, w: f64, h: f64) {
  turtle.set_colour("black");
  let x = neuron.x() as f64 * w - w*0.5 + x_offset;
  let y = neuron.y() as f64 * h - h*0.5 + y_offset;
  turtle.draw_dot(x, y, 5.0);
}

fn draw_all_neurons(turtle: &mut THelper, network_info: &NetworkInfo, x_o: f64, y_o: f64, w: f64, h: f64) {
  for n in network_info.neurons() {
    draw_neuron(turtle, n, x_o, y_o, w, h);
  }
}

fn draw_genes(turtle: &mut THelper, genome: &Genome, 
                     x_offset: f64, y_offset: f64, w: f64, h: f64) {
  
  let neurons = genome.network().neurons().clone();
  let genes = genome.genes();
  
  for con in genes {
    let from = con.from();
    let to = con.to();
    
    let n_from = {
      let mut n = neurons[0].clone();
      for i in 0..neurons.len() {
        if neurons[i].id() == from {
          n = neurons[i].clone();
          break;
        }
      }
      n
    };
    let n_to = {
      let mut n = neurons[0].clone();
      for i in 0..neurons.len() {
        if neurons[i].id() == to {
          n = neurons[i].clone();
          break;
        }
      }
      n
    };
    
    //if let Some(n_from) = network_info.get_neuron_by_id(from) {
     // if let Some(n_to) = network_info.get_neuron_by_id(to) {
        let x = n_from.x() as f64 * w - w*0.5 + x_offset;
        let y = n_from.y() as f64 * h - h*0.5 + y_offset;
        let x1 = n_to.x() as f64 * w - w*0.5 + x_offset;
        let y1 = n_to.y() as f64 * h - h*0.5 + y_offset;
        
        if con.is_enabled() {
          turtle.set_colour("green");
        } else {
          turtle.set_colour("red");
        }
        
        let weight = con.weight();
        
        let pen_size = ((weight + 2.0) * 2.0).max(0.001);
        turtle.pen_size(pen_size);
        
        turtle.draw_line(x, y, x1, y1);
        turtle.pen_size(1.0);
     // }
   // }
  }
}

pub fn draw_genome(turtle: &mut THelper, genome: &Genome, x: f64, y: f64, w: f64, h: f64) {
  draw_genes(turtle, genome, x, y, w, h);
  for n in &genome.network_neurons() {
    draw_neuron(turtle, n, x, y, w, h);
  }
}

fn main() {
  println!("Hello, world!");
  let mut rng = thread_rng();
  
  let w = 800.0;
  let h = 600.0;
  let speed = 25;
  let mut turtle = THelper::new(w, h, speed);
  //turtle.set_speed("instant");
  let mut population = 300;
  
  let inputs = 2;
  let outputs = 1;
  
  let mut params = NeatParams::new();
  let mut network_info = NetworkInfo::new(inputs, outputs, &params);
  let mut pool = Pool::new(inputs, outputs, population);
  
  pool.initialise_pool(&params, &mut network_info, &mut rng);
  
  turtle.wait();
  
  //test_load_save_genome(population, w, h, &mut turtle, &mut pool, &mut network_info, &mut params, &mut rng);
  XOR_test(population, w, h, &mut turtle, &mut pool, &mut network_info, &mut params, &mut rng);
  //repeat_run_test(population, w, h, &mut turtle, &mut pool, &mut network_info, &mut params, &mut rng);
  
  /*
  let mut basic = Genome::basic(inputs+1, outputs, &params, &mut network_info, &mut rng);
  while basic.genes().len() == 0 {
    basic.link_mutate(false, &params, &mut network_info, &mut rng);
  }
  loop {
    turtle.wait();
    basic.node_mutate(&mut network_info, &mut rng);
    turtle.clear();
    
    basic.generate_network(&params, &network_info);
    println!("Output: {:?}", basic.evaluate_network(vec![1.0, 1.0, 1.0], &params, &network_info));
    
    draw_genome(&mut turtle, &basic, 0.0, 0.0, w, h);
    /*draw_genome_genes(&mut turtle, basic.genes(), &network_info, 0.0, 0.0, w, h);
    println!("{:?}", basic.genes());
    for n in &basic.network_neurons() {
      draw_neuron(&mut turtle, n, 0.0, 0.0, w, h);
      println!("{:?}", n);
    }*/
    turtle.wait();
    turtle.clear();
    //basic.reposition_neurons(&mut rng);
    draw_genome(&mut turtle, &basic, 0.0, 0.0, w, h);
    //basic.mutate(&params, &mut network_info, &mut rng);
  }*/
  /*
  loop {
    let mut best_fitness = -99999999999999.9;
    let mut worst_fitness = 9999999999999.9;
    let mut best_genome = pool.current_genome().clone();
    let mut worst_genome = pool.current_genome().clone();
    
    for i in 0..population {
      turtle.clear();
      let output = pool.evaluate_current(vec![0.5, 0.5], &params, &network_info);
      pool.set_fitness_current(output.iter().sum());
      println!("Outputs: {:?}", output);
      
      if pool.current_genome().fitness() > best_fitness {
        best_fitness = pool.current_genome().fitness();
        best_genome = pool.current_genome().clone();
      }
      if pool.current_genome().fitness() < worst_fitness {
        worst_fitness = pool.current_genome().fitness();
        worst_genome = pool.current_genome().clone();
      }
      
      pool.next_genome(&mut network_info, &params, &mut rng);
    }
    
    //draw_all_neurons(&mut turtle, &network_info, w*-0.25, 0.0, w*0.5, h);
    draw_genome_genes(&mut turtle, best_genome.genes(), &network_info, 0.0, 0.0, w, h);
    for n in &best_genome.network_neurons() {
      draw_neuron(&mut turtle, n, 0.0, 0.0, w, h);
    }
    /*
    draw_all_neurons(&mut turtle, &network_info, w*0.25, 0.0, w*0.5, h);
    draw_genome_genes(&mut turtle, worst_genome.genes(), &network_info, w*0.25, 0.0, w*0.5, h);*/
    turtle.wait();
  }*/
 
}
