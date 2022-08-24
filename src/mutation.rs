use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::rc::Rc;
extern crate test;

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

#[derive(Default)]
struct SOAMutation2 {
    mutations: Vec<Mutation2>,
    counts: Vec<u32>,
}

#[derive(Default)]
struct RcMutation {
    mutations: Vec<Rc<Mutation2>>,
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

#[derive(Default)]
struct RcGenomes {
    mutations: Vec<Rc<Mutation2>>,
    offsets: Vec<usize>,
}

impl RcGenomes {
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
    mutation_queue: Vec<usize>,
    alive_genomes: DODGenomes,
    offspring_genomes: DODGenomes,
    rng: StdRng,
    genome_length: f64,
    popsize: u32,
}

impl DODPopulation {
    // HACK: ::default() but not public
    fn quick() -> Self {
        Self {
            mutations: DODmutations::default(),
            mutation_queue: vec![],
            alive_genomes: DODGenomes::default(),
            offspring_genomes: DODGenomes::default(),
            rng: StdRng::seed_from_u64(0),
            genome_length: 1e9,
            popsize: 1000,
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

        //println!("{}", 1.0 / mutation_rate / self.genome_length);
        let positionator = rand_distr::Exp::new(mutation_rate).unwrap();

        let mut last_mutation_pos = self.rng.sample(positionator);

        let mut parent_genome_index = 0_usize;
        let mut nmuts = 0;
        while last_mutation_pos < self.genome_length {
            //println!("{}", last_mutation_pos);
            while parent_genome_index < parent_genome.len()
                && self.mutations.position[parent_genome[parent_genome_index]] <= last_mutation_pos
            {
                self.offspring_genomes
                    .mutations
                    .push(parent_genome[parent_genome_index]);
                parent_genome_index += 1;
            }
            // Add new mutation -- this should be a "callback"/trait object
            let new_mutation_index = match self.mutation_queue.pop() {
                Some(index) => {
                    self.mutations.position[index] = last_mutation_pos;
                    self.mutations.count[index] = 0;
                    self.mutations.effect_size[index] = 0.0;
                    index
                }
                None => {
                    self.mutations.position.push(last_mutation_pos);
                    self.mutations.count.push(0);
                    self.mutations.effect_size.push(0.0);
                    self.mutations.position.len() - 1
                }
            };
            self.offspring_genomes.mutations.push(new_mutation_index);
            last_mutation_pos += self.rng.sample(positionator);
            nmuts += 1;
        }
        // println!("{}", nmuts);

        for i in parent_genome_index..parent_genome.len() {
            self.offspring_genomes.mutations.push(parent_genome[i]);
        }
        self.offspring_genomes.offsets.push(next_alive_offset);

        #[cfg(debug_assertions)]
        {
            let sorted = self.offspring_genomes.mutations[next_alive_offset..]
                .windows(2)
                .all(|s| self.mutations.position[s[0]] <= self.mutations.position[s[1]]);
            assert!(sorted);
        }
        //println!("{:?}", self.offspring_genomes);
    }

    fn swap_generations(&mut self) {
        std::mem::swap(&mut self.alive_genomes, &mut self.offspring_genomes);
        self.offspring_genomes.clear();
    }
}

struct RcPopulation {
    mutations: RcMutation,
    mutation_queue: Vec<usize>,
    alive_genomes: RcGenomes,
    offspring_genomes: RcGenomes,
    rng: StdRng,
    genome_length: f64,
    popsize: u32,
}

impl RcPopulation {
    // HACK: ::default() but not public
    fn quick() -> Self {
        Self {
            mutations: RcMutation::default(),
            mutation_queue: vec![],
            alive_genomes: RcGenomes::default(),
            offspring_genomes: RcGenomes::default(),
            rng: StdRng::seed_from_u64(0),
            genome_length: 1e9,
            popsize: 1000,
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

        //println!("{}", 1.0 / mutation_rate / self.genome_length);
        let positionator = rand_distr::Exp::new(mutation_rate).unwrap();

        let mut last_mutation_pos = self.rng.sample(positionator);

        let mut parent_genome_index = 0_usize;
        let mut nmuts = 0;
        while last_mutation_pos < self.genome_length {
            //println!("{}", last_mutation_pos);
            while parent_genome_index < parent_genome.len()
                && parent_genome[parent_genome_index].position <= last_mutation_pos
            {
                self.offspring_genomes
                    .mutations
                    .push(parent_genome[parent_genome_index].clone());
                parent_genome_index += 1;
            }
            // Add new mutation -- this should be a "callback"/trait object
            let m = Rc::new(Mutation2 {
                effect_size: 0.0,
                position: last_mutation_pos,
            });
            match self.mutation_queue.pop() {
                Some(index) => {
                    self.mutations.mutations[index] = m.clone();
                }
                None => {
                    self.mutations.mutations.push(m.clone());
                }
            }
            self.offspring_genomes.mutations.push(m);
            last_mutation_pos += self.rng.sample(positionator);
            nmuts += 1;
        }
        // println!("{}", nmuts);

        for i in parent_genome_index..parent_genome.len() {
            self.offspring_genomes
                .mutations
                .push(parent_genome[i].clone());
        }
        self.offspring_genomes.offsets.push(next_alive_offset);

        #[cfg(debug_assertions)]
        {
            let sorted = self.offspring_genomes.mutations[next_alive_offset..]
                .windows(2)
                .all(|s| s[0].position <= s[1].position);
            assert!(sorted);
        }
        //println!("{:?}", self.offspring_genomes);
    }

    fn swap_generations(&mut self) {
        std::mem::swap(&mut self.alive_genomes, &mut self.offspring_genomes);
        self.offspring_genomes.clear();
    }
}

struct SOAPopulation2 {
    mutations: SOAMutation2,
    mutation_queue: Vec<usize>,
    alive_genomes: DODGenomes,
    offspring_genomes: DODGenomes,
    rng: StdRng,
    genome_length: f64,
    popsize: u32,
}

impl SOAPopulation2 {
    // HACK: ::default() but not public
    fn quick() -> Self {
        Self {
            mutations: SOAMutation2::default(),
            mutation_queue: vec![],
            alive_genomes: DODGenomes::default(),
            offspring_genomes: DODGenomes::default(),
            rng: StdRng::seed_from_u64(0),
            genome_length: 1e9,
            popsize: 1000,
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

        //println!("{}", 1.0 / mutation_rate / self.genome_length);
        let positionator = rand_distr::Exp::new(mutation_rate).unwrap();

        let mut last_mutation_pos = self.rng.sample(positionator);

        let mut parent_genome_index = 0_usize;
        while last_mutation_pos < self.genome_length {
            //println!("{}", last_mutation_pos);
            while parent_genome_index < parent_genome.len()
                && self.mutations.mutations[parent_genome[parent_genome_index]].position
                    <= last_mutation_pos
            {
                self.offspring_genomes
                    .mutations
                    .push(parent_genome[parent_genome_index]);
                parent_genome_index += 1;
            }
            // Add new mutation -- this should be a "callback"/trait object
            let new_mutation_index = match self.mutation_queue.pop() {
                Some(index) => {
                    self.mutations.mutations[index].position = last_mutation_pos;
                    self.mutations.mutations[index].effect_size = 0.0;
                    self.mutations.counts[index] = 0;
                    index
                }
                None => {
                    let mutation = Mutation2 {
                        position: last_mutation_pos,
                        effect_size: 0.0,
                    };
                    self.mutations.mutations.push(mutation);
                    self.mutations.counts.push(0);
                    self.mutations.mutations.len() - 1
                }
            };
            self.offspring_genomes.mutations.push(new_mutation_index);
            last_mutation_pos += self.rng.sample(positionator);
        }
        // println!("{}", nmuts);

        for i in parent_genome_index..parent_genome.len() {
            self.offspring_genomes.mutations.push(parent_genome[i]);
        }
        self.offspring_genomes.offsets.push(next_alive_offset);

        #[cfg(debug_assertions)]
        {
            let sorted = self.offspring_genomes.mutations[next_alive_offset..]
                .windows(2)
                .all(|s| {
                    self.mutations.mutations[s[0]].position
                        <= self.mutations.mutations[s[1]].position
                });
            assert!(sorted);
        }
        //println!("{:?}", self.offspring_genomes);
    }

    fn swap_generations(&mut self) {
        std::mem::swap(&mut self.alive_genomes, &mut self.offspring_genomes);
        self.offspring_genomes.clear();
    }
}

trait GenerateBirths {
    fn births(&mut self, mutation_rate: f64);
}

trait StartGeneration {
    fn start(&mut self);
}

trait FinishGeneration {
    fn finish(&mut self);
}

impl GenerateBirths for DODPopulation {
    fn births(&mut self, mutation_rate: f64) {
        let u = rand_distr::Uniform::new(0, self.popsize);
        for _ in 0..self.popsize {
            let parent = self.rng.sample(u);
            self.generate_offspring(parent as usize, mutation_rate);
        }
    }
}

impl StartGeneration for DODPopulation {
    fn start(&mut self) {
        self.mutation_queue.clear();
        self.mutations.count.iter().enumerate().for_each(|(i, c)| {
            if *c == 0 {
                self.mutation_queue.push(i);
            }
        });
        self.mutations.count.fill(0);
        //println!(
        //    "{} {}",
        //    self.mutation_queue.len(),
        //    self.mutations.count.len()
        //);
    }
}

impl FinishGeneration for DODPopulation {
    fn finish(&mut self) {
        self.swap_generations();
        self.alive_genomes
            .mutations
            .iter()
            .for_each(|m| self.mutations.count[*m] += 1);
    }
}

impl GenerateBirths for RcPopulation {
    fn births(&mut self, mutation_rate: f64) {
        let u = rand_distr::Uniform::new(0, self.popsize);
        for _ in 0..self.popsize {
            let parent = self.rng.sample(u);
            self.generate_offspring(parent as usize, mutation_rate);
        }
    }
}

impl StartGeneration for RcPopulation {
    fn start(&mut self) {
        self.mutation_queue.clear();
        for (i, m) in self.mutations.mutations.iter().enumerate() {
            if Rc::strong_count(m) == 1 {
                self.mutation_queue.push(i);
            }
        }
    }
}

impl FinishGeneration for RcPopulation {
    fn finish(&mut self) {
        self.swap_generations();
    }
}

impl GenerateBirths for SOAPopulation2 {
    fn births(&mut self, mutation_rate: f64) {
        let u = rand_distr::Uniform::new(0, self.popsize);
        for _ in 0..self.popsize {
            let parent = self.rng.sample(u);
            self.generate_offspring(parent as usize, mutation_rate);
        }
    }
}

impl StartGeneration for SOAPopulation2 {
    fn start(&mut self) {
        self.mutation_queue.clear();
        self.mutations.counts.iter().enumerate().for_each(|(i, c)| {
            if *c == 0 {
                self.mutation_queue.push(i);
            }
        });
        self.mutations.counts.fill(0);
    }
}

impl FinishGeneration for SOAPopulation2 {
    fn finish(&mut self) {
        self.swap_generations();
        self.alive_genomes
            .mutations
            .iter()
            .for_each(|m| self.mutations.counts[*m] += 1);
    }
}

fn evolve<P>(ngenerations: u32, mutation_rate: f64, pop: &mut P)
where
    P: GenerateBirths + FinishGeneration + StartGeneration,
{
    for _ in 0..ngenerations {
        pop.start();
        pop.births(mutation_rate);
        pop.finish();
    }
}

#[cfg(test)]
mod test_mutation_concepts {
    use super::*;
    use test::Bencher;

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

    #[test]
    fn test_evolve_dod() {
        let mut pop = DODPopulation::quick();
        evolve(10, 0.5e-9, &mut pop);
        println!(
            "{} {} {}",
            pop.alive_genomes.mutations.len(),
            pop.mutations.count.len(),
            pop.mutations.count.iter().filter(|c| **c > 0).count()
        );

        // mutations cannot exist more times than there are alive individuals
        assert!(pop.mutations.count.iter().all(|c| *c <= pop.popsize));
        for (i, o) in pop.alive_genomes.offsets.iter().enumerate() {
            let genome = if i < pop.alive_genomes.offsets.len() {
                if i + 1 < pop.alive_genomes.offsets.len() {
                    &pop.alive_genomes.mutations[*o..pop.alive_genomes.offsets[i + 1]]
                } else {
                    &pop.alive_genomes.mutations[*o..]
                }
            } else {
                &pop.alive_genomes.mutations[*o..]
            };
            // an alive genome cannot contain an extinct variant
            assert!(genome.iter().all(|m| pop.mutations.count[*m] > 0));
            // genomes must contain mutations sorted by position.
            let sorted = genome
                .windows(2)
                .all(|s| pop.mutations.position[s[0]] <= pop.mutations.position[s[1]]);
            assert!(sorted, "failed on genome {}", i); //, "{:?}", genome);
        }
    }

    #[test]
    fn test_evolve_rc() {
        let mut pop = RcPopulation::quick();
        evolve(10, 0.5e-9, &mut pop);
        println!(
            "{} {}",
            pop.alive_genomes.mutations.len(),
            pop.mutations
                .mutations
                .iter()
                .filter(|m| Rc::strong_count(m) > 1)
                .count()
        );
    }

    #[bench]
    fn bench_evolve_dod_high_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = DODPopulation::quick();
            let mutrate = 0.5 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_dod_low_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = DODPopulation::quick();
            let mutrate = 0.1 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_dod_very_low_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = DODPopulation::quick();
            let mutrate = 1e-4 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_soa2_high_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = SOAPopulation2::quick();
            let mutrate = 0.5 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_soa2_low_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = SOAPopulation2::quick();
            let mutrate = 0.1 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_soa2_very_low_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = SOAPopulation2::quick();
            let mutrate = 1e-4 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_rc_high_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = RcPopulation::quick();
            let mutrate = 0.5 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_rc_low_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = RcPopulation::quick();
            let mutrate = 0.1 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }

    #[bench]
    fn bench_evolve_rc_very_low_mutrate(b: &mut Bencher) {
        b.iter(|| {
            let mut pop = RcPopulation::quick();
            let mutrate = 1e-4 / pop.genome_length;
            evolve(500, mutrate, &mut pop);
        });
    }
}
