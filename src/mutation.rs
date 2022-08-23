trait IncrementCount {
    fn increment(&mut self, index: usize);
}

struct DODmutations {
    count: Vec<u32>,
    effect_size: Vec<f64>,
}

struct Mutation {
    count: u32,
    effect_size: f64,
}

struct Mutation2 {
    effect_size: f64,
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

struct DODGenomes {
    mutations: Vec<usize>,
    offsets: Vec<usize>,
}

// This is totally the wrong concept
trait Mutate {
    // Return number of mutations
    fn mutate(individual: usize, rate: f64) -> u32;
}

struct DODPopulation {
    mutations: DODmutations,
    genomes: DODGenomes,
}
