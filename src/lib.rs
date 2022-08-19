// We want to store things as
// arrays, but treat "table rows"
// as object-like via traits.
pub struct AliveIndividuals {
    genetic_value: Vec<f64>,
    nodes: Vec<i32>,
    ploidy: usize,
    genetic_value_stride: usize,
}

impl AliveIndividuals {
    pub fn new(ploidy: usize, genetic_value_stride: usize) -> Self {
        assert!(ploidy > 0);
        assert!(genetic_value_stride > 0);
        Self {
            genetic_value: vec![],
            nodes: vec![],
            ploidy,
            genetic_value_stride,
        }
    }

    pub fn add_individual(&mut self, genetic_value: &[f64], nodes: &[i32]) {
        assert_eq!(genetic_value.len(), self.genetic_value_stride);
        assert_eq!(nodes.len(), self.ploidy);
        self.genetic_value.extend_from_slice(genetic_value);
        self.nodes.extend_from_slice(nodes);
    }
}

// Not best design:
// * implies panic if fail, etc..
//
//On the plus side
// * These are population-level traits,
//   independent of struct of array or
//   array of struct idioms
// * We are just asking about the i-th individual!

trait IndividualGeneticValues {
    fn genetic_values(&self, individual: usize) -> &[f64];
}

trait IndividualNodes {
    fn nodes(&self, individual: usize) -> &[i32];
}

impl IndividualGeneticValues for AliveIndividuals {
    fn genetic_values(&self, individual: usize) -> &[f64] {
        &self.genetic_value[individual * self.genetic_value_stride
            ..individual * self.genetic_value_stride + self.genetic_value_stride]
    }
}

impl IndividualNodes for AliveIndividuals {
    fn nodes(&self, individual: usize) -> &[i32] {
        &self.nodes[individual * self.ploidy..individual * self.ploidy + self.ploidy]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_fetch() {
        let mut individuals = AliveIndividuals::new(2, 1);
        individuals.add_individual(&[1.0], &[1, 2]);

        assert_eq!(individuals.genetic_values(0), &[1.0]);
        assert_eq!(individuals.nodes(0), &[1, 2]);
    }
}
