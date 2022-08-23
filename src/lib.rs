mod mutation;

// We want to store things as
// arrays, but treat "table rows"
// as object-like via traits.
pub struct DODIndividualMetadata {
    genetic_value: Vec<f64>,
    nodes: Vec<i32>,
    ploidy: usize,
    genetic_value_stride: usize,
}

impl DODIndividualMetadata {
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

struct GeneticValueIterator<'alive> {
    alive: &'alive DODIndividualMetadata,
    offset: usize,
}

impl<'alive> Iterator for GeneticValueIterator<'alive> {
    type Item = &'alive [f64];

    fn next(&mut self) -> Option<Self::Item> {
        // NOTE: may not be the most rigorous check
        if self.offset * self.alive.genetic_value_stride + self.alive.genetic_value_stride
            <= self.alive.genetic_value.len()
        {
            let i = self.offset;
            self.offset += 1;
            Some(self.alive.genetic_values(i))
        } else {
            None
        }
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

trait IterateGeneticValues<'alive> {
    type Output: Sized + Iterator<Item = &'alive [f64]>;
    fn genetic_value_iterator(&'alive self) -> Self::Output;
}

impl IndividualGeneticValues for DODIndividualMetadata {
    fn genetic_values(&self, individual: usize) -> &[f64] {
        &self.genetic_value[individual * self.genetic_value_stride
            ..individual * self.genetic_value_stride + self.genetic_value_stride]
    }
}

impl IndividualNodes for DODIndividualMetadata {
    fn nodes(&self, individual: usize) -> &[i32] {
        &self.nodes[individual * self.ploidy..individual * self.ploidy + self.ploidy]
    }
}

impl<'alive> IterateGeneticValues<'alive> for DODIndividualMetadata {
    type Output = GeneticValueIterator<'alive>;
    fn genetic_value_iterator(&'alive self) -> Self::Output {
        GeneticValueIterator {
            alive: self,
            offset: 0,
        }
    }
}

trait AsGeneticValueIterator {
    fn as_genetic_value_iterator(&self) -> Box<dyn Iterator<Item = &[f64]> + '_>;
}

impl AsGeneticValueIterator for DODIndividualMetadata {
    fn as_genetic_value_iterator(&self) -> Box<dyn Iterator<Item = &[f64]> + '_> {
        Box::new(GeneticValueIterator {
            alive: self,
            offset: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_fetch() {
        let mut individuals = DODIndividualMetadata::new(2, 1);
        individuals.add_individual(&[1.0], &[1, 2]);

        let x = &individuals as &dyn IterateGeneticValues<Output = GeneticValueIterator>;

        assert_eq!(individuals.genetic_values(0), &[1.0]);
        assert_eq!(individuals.nodes(0), &[1, 2]);

        let iter = GeneticValueIterator {
            alive: &individuals,
            offset: 0,
        };
        for i in iter {
            assert_eq!(i, &[1.0]);
        }

        assert_eq!(individuals.genetic_value_iterator().count(), 1);
    }

    // This is still not dynamic dispatch due to generics.
    // It is a hybrid mish-mash, which may be okay.
    fn iterate<'alive, T: Sized + Iterator<Item = &'alive [f64]>>(
        x: &'alive dyn IterateGeneticValues<'alive, Output = T>,
    ) {
        for i in x.genetic_value_iterator() {
            assert_eq!(i, &[1.0]);
        }
    }

    //fn iterate<'alive, I, T>(x: &'alive I)
    //where
    //    I: IterateGeneticValues<'alive, Output = T>,
    //    T: Sized + Iterator<Item = &'alive [f64]>,
    //{
    //    for i in x.genetic_value_iterator() {
    //        assert_eq!(i, &[1.0]);
    //    }
    //}

    #[test]
    fn test_object_safety() {
        let mut individuals = DODIndividualMetadata::new(2, 1);
        individuals.add_individual(&[1.0], &[1, 2]);

        let x = &individuals as &dyn IterateGeneticValues<Output = GeneticValueIterator>;
        iterate(&individuals);
        iterate(x);

        for i in individuals.as_genetic_value_iterator() {
            assert_eq!(i, &[1.0]);
        }
    }
}
