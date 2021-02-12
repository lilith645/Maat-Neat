#[macro_use]
pub extern crate serde_derive;
pub extern crate bincode;

pub use self::modules::{Pool, NetworkInfo, NeatParams, Network, Species, Genome, Neuron, NodeType};

mod modules;

#[cfg(test)]
mod tests {
  use super::*;
  
}
