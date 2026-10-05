// Command go-agent-runner is a standalone version of the trial loop from the
// post: run each task k times against a fresh in-memory cluster, then report
// pass@k and pass^k. Stdlib only. Run with `go run .`.
//
// The agent is scripted with a per-task failure rate, so you can watch the two
// metrics diverge as k grows: pass@k climbs toward 1 (retries hide flakiness)
// while pass^k falls toward 0 (an agent with write access must be right every
// time, not once in k).
package main

import (
    "flag"
    "fmt"
    "math/rand/v2"
    "os"
    "strings"
    "text/tabwriter"
)

// Deployment is the only kind of object in the toy cluster.
type Deployment struct {
    Name      string
    Namespace string
    MemoryMi  int
    NeedsMi   int // ground truth the agent cannot see
    Status    string
}

// Cluster holds state for exactly one trial. It is never shared.
type Cluster struct {
    Deployments map[string]*Deployment
    Writes      []string
}

// NewCluster copies the seed (Go 1.22+ gives each loop iteration its own d),
// so no trial can see another trial's writes.
func NewCluster(seed []Deployment) *Cluster {
    c := &Cluster{Deployments: map[string]*Deployment{}}
    for _, d := range seed {
        c.Deployments[d.Namespace+"/"+d.Name] = &d
    }
    c.reconcile()
    return c
}

// reconcile derives status from desired state, like a controller loop.
func (c *Cluster) reconcile() {
    for _, d := range c.Deployments {
        if d.MemoryMi < d.NeedsMi {
            d.Status = "CrashLoopBackOff"
        } else {
            d.Status = "Running"
        }
    }
}

func (c *Cluster) SetMemory(key string, mi int) {
    c.Writes = append(c.Writes, fmt.Sprintf("set_memory %s %dMi", key, mi))
    c.Deployments[key].MemoryMi = mi
    c.reconcile()
}

func (c *Cluster) Restart(key string) {
    c.Writes = append(c.Writes, "restart "+key)
    c.reconcile() // a restart does not change the root cause
}

// Task is one incident: a seed, a target, and a check on the final state.
type Task struct {
    ID          string
    Target      string
    Seed        []Deployment
    FailureRate float64 // how often the scripted agent takes the wrong action
}

// Passed grades the outcome and an invariant, never the sequence of actions.
func (t Task) Passed(c *Cluster) bool {
    if c.Deployments[t.Target].Status != "Running" {
        return false
    }
    ns := strings.SplitN(t.Target, "/", 2)[0]
    for _, w := range c.Writes {
        if !strings.Contains(w, " "+ns+"/") {
            return false // wrote outside the incident's namespace
        }
    }
    return true
}

// Agent acts on a cluster. This one is scripted: usually right, sometimes not.
type Agent func(rng *rand.Rand, t Task, c *Cluster)

func ScriptedAgent(rng *rand.Rand, t Task, c *Cluster) {
    if rng.Float64() < t.FailureRate {
        c.Restart(t.Target) // the classic wrong fix
        return
    }
    d := c.Deployments[t.Target]
    c.SetMemory(t.Target, d.MemoryMi*2)
}

// Trial is the record of one run of one task.
type Trial struct {
    TaskID string
    Index  int
    Passed bool
    Writes []string
}

// RunTask runs a task n times, each against a freshly seeded cluster.
func RunTask(t Task, agent Agent, n int, rng *rand.Rand) []Trial {
    trials := make([]Trial, 0, n)
    for i := 0; i < n; i++ {
        c := NewCluster(t.Seed)
        agent(rng, t, c)
        trials = append(trials, Trial{TaskID: t.ID, Index: i, Passed: t.Passed(c), Writes: c.Writes})
    }
    return trials
}

// PassAtK is the unbiased estimate of P(at least one of k samples passes)
// given c passes in n trials: 1 - C(n-c, k) / C(n, k).
func PassAtK(n, c, k int) float64 {
    if n-c < k {
        return 1.0
    }
    return 1.0 - ratio(n-c, n, k)
}

// PassHatK estimates P(all k samples pass): C(c, k) / C(n, k).
func PassHatK(n, c, k int) float64 {
    if c < k {
        return 0.0
    }
    return ratio(c, n, k)
}

// ratio computes C(a, k) / C(n, k) as a running product to avoid overflow.
func ratio(a, n, k int) float64 {
    r := 1.0
    for i := 0; i < k; i++ {
        r *= float64(a-i) / float64(n-i)
    }
    return r
}

func main() {
    trials := flag.Int("trials", 20, "trials per task (must be >= the largest k)")
    seed := flag.Uint64("seed", 7, "random seed, so runs are reproducible")
    flag.Parse()

    tasks := []Task{
        {ID: "oomkilled-api", Target: "shop/checkout-api", FailureRate: 0.05,
            Seed: []Deployment{
                {Name: "checkout-api", Namespace: "shop", MemoryMi: 128, NeedsMi: 256},
                {Name: "payments", Namespace: "billing", MemoryMi: 256, NeedsMi: 128},
            }},
        {ID: "oomkilled-worker", Target: "jobs/report-worker", FailureRate: 0.15,
            Seed: []Deployment{{Name: "report-worker", Namespace: "jobs", MemoryMi: 256, NeedsMi: 512}}},
        {ID: "oomkilled-cache", Target: "edge/cache", FailureRate: 0.30,
            Seed: []Deployment{{Name: "cache", Namespace: "edge", MemoryMi: 512, NeedsMi: 1024}}},
    }
    ks := []int{1, 3, 5, 10}
    if *trials < ks[len(ks)-1] {
        fmt.Fprintf(os.Stderr, "need -trials >= %d\n", ks[len(ks)-1])
        os.Exit(2)
    }

    rng := rand.New(rand.NewPCG(*seed, *seed))
    w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
    fmt.Fprintf(w, "task\tfail rate\tpassed\t")
    for _, k := range ks {
        fmt.Fprintf(w, "pass@%d\tpass^%d\t", k, k)
    }
    fmt.Fprintln(w)

    sumAt := make([]float64, len(ks))
    sumHat := make([]float64, len(ks))
    for _, t := range tasks {
        results := RunTask(t, ScriptedAgent, *trials, rng)
        c := 0
        for _, r := range results {
            if r.Passed {
                c++
            }
        }
        fmt.Fprintf(w, "%s\t%.2f\t%d/%d\t", t.ID, t.FailureRate, c, *trials)
        for i, k := range ks {
            at, hat := PassAtK(*trials, c, k), PassHatK(*trials, c, k)
            sumAt[i] += at
            sumHat[i] += hat
            fmt.Fprintf(w, "%.2f\t%.2f\t", at, hat)
        }
        fmt.Fprintln(w)
    }
    fmt.Fprintf(w, "mean\t\t\t")
    for i := range ks {
        n := float64(len(tasks))
        fmt.Fprintf(w, "%.2f\t%.2f\t", sumAt[i]/n, sumHat[i]/n)
    }
    fmt.Fprintln(w)
    w.Flush()

    fmt.Println()
    fmt.Println("pass@k rises with k: given enough retries, something works.")
    fmt.Println("pass^k falls with k: the chance that every one of k runs works.")
    fmt.Println("For an agent that writes to production, pass^k is the number to watch.")
}
