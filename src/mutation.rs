use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

trait IncrementCount {
    fn increment(&mut self, index: usize);
}

#[derive(Default)]
struct DODmutations {
    count: Vec<u32>,
    effect_size: Vec<f64>,
    position: Vec<f64>,
}

impl DODmutations {
    fn is_empty(&self) -> bool {
        assert_eq!(self.count.len(), self.effect_size.len());
        assert_eq!(self.count.len(), self.position.len());
        self.count.is_empty()
    }
}

struct Mutation {
    count: u32,
    effect_size: f64,
    position: f64,
}

struct Mutation2 {
    effect_size: f64,
    position: f64,
}

struct SOAMutation {
    mutations: Vec<Mutation>,
}

struct SOAMutation2 {
    mutations: Vec<Mutation2>,
    counts: Vec<u32>,
}

impl IncrementCount for DODmutations {
    fn increment(&mut self, index: usize) {
        self.count[index] += 1;
    }
}

impl IncrementCount for SOAMutation {
    fn increment(&mut self, index: usize) {
        self.mutations[index].count += 1;
    }
}

impl IncrementCount for SOAMutation2 {
    fn increment(&mut self, index: usize) {
        self.counts[index] += 1;
    }
}

#[derive(Default, Debug)]
struct DODGenomes {
    mutations: Vec<usize>,
    offsets: Vec<usize>,
}

impl DODGenomes {
    fn clear(&mut self) {
        self.mutations.clear();
        self.offsets.clear();
    }
}

// This is totally the wrong concept
trait Mutate {
    // Return number of mutations
    fn mutate(individual: usize, rate: f64) -> u32;
}

struct DODPopulation {
    mutations: DODmutations,
    alive_genomes: DODGenomes,
    offspring_genomes: DODGenomes,
    rng: StdRng,
    genome_length: f64,
}

impl DODPopulation {
    fn quick() -> Self {
        Self {
            mutations: DODmutations::default(),
            alive_genomes: DODGenomes::default(),
            offspring_genomes: DODGenomes::default(),
            rng: StdRng::seed_from_u64(0),
            genome_length: 1e9,
        }
    }
    fn generate_offspring(&mut self, parent: usize, mutation_rate: f64) {
        // NOTE: this is really terrible
        let parent_genome = if parent < self.alive_genomes.offsets.len() {
            if parent + 1 < self.alive_genomes.offsets.len() {
                &self.alive_genomes.mutations
                    [self.alive_genomes.offsets[parent]..self.alive_genomes.offsets[parent + 1]]
            } else {
                {
                    &self.alive_genomes.mutations[self.alive_genomes.offsets[parent]..]
                }
            }
        } else {
            if !self.alive_genomes.mutations.is_empty() {
                &self.alive_genomes.mutations[self.alive_genomes.offsets[parent]..]
            } else {
                &[]
            }
        };
        let next_alive_offset = self.offspring_genomes.mutations.len();

        println!("{}", 1.0 / mutation_rate / self.genome_length);
        let positionator = rand_distr::Exp::new(mutation_rate).unwrap();

        let mut last_mutation_pos = self.rng.sample(positionator);

        let mut parent_genome_index = 0_usize;
        while last_mutation_pos < self.genome_length {
            println!("{}", last_mutation_pos);
            while parent_genome_index < parent_genome.len()
                && self.mutations.position[parent_genome[parent_genome_index]] <= last_mutation_pos
            {
                self.offspring_genomes
                    .mutations
                    .push(parent_genome[parent_genome_index]);
                parent_genome_index += 1;
            }
            // Add new mutation
            self.mutations.position.push(last_mutation_pos);
            self.mutations.count.push(0);
            self.mutations.effect_size.push(0.0);
            self.offspring_genomes
                .mutations
                .push(self.mutations.position.len() - 1);
            last_mutation_pos += self.rng.sample(positionator);
        }

        for i in parent_genome_index..parent_genome.len() {
            self.offspring_genomes.mutations.push(parent_genome[i]);
        }
        self.offspring_genomes.offsets.push(next_alive_offset);
        println!("{:?}", self.offspring_genomes);
    }

    fn swap_generations(&mut self) {
        std::mem::swap(&mut self.alive_genomes, &mut self.offspring_genomes);
        self.offspring_genomes.clear();
    }
}

#[cfg(test)]
mod test_mutation_concepts {
    use super::*;

    #[test]
    fn quick_test() {
        let mut pop = DODPopulation::quick();
        // HACK: this is bad design
        // pop.alive_genomes.offsets.push(0); // There is an individual w/no mutations
        pop.generate_offspring(0, 1e-8);
        assert!(!pop.mutations.is_empty());
        pop.swap_generations();
        pop.generate_offspring(0, 1e-8);

        // NOTE: there is only one genome, so we can do the following hack:
        let sorted = pop
            .alive_genomes
            .mutations
            .windows(2)
            .all(|s| pop.mutations.position[s[0]] <= pop.mutations.position[s[1]]);
        assert!(sorted);

        // NOTE: we should be in an okay position
        // for counts
        pop.alive_genomes
            .mutations
            .iter()
            .for_each(|m| pop.mutations.count[*m] += 1);
    }
}
