# TODO
## 1. tighten up the ship
### a. update file structure to be nicer e.g.
```
projects/
  MyProject/
    runs/
      run_3_20250321_153055/
        configs/
          config_esr_1.json  (made by the agent)
        data/
          data_esr_1.json
          plot_esr_1.png
          data_find_nv_1.json
          plot_find_nv_1.png
          ...
        logs/
          log_3.txt
```
- This will lend itself to being more of a structured database that the agent can knowledge-extract from (e.g. RAG, another action that involves reading from the database structure, etc.)
    - Becomes a dynamic database and long-term memory almost? Especially w/ human supervision
    - No longer starting from scratch W, everything is saved into a log
        - But it has to be more efficient...first glean fundamental rules, then selectively do logs based on similarity to present experiment
    - *improves decision-making*

### b. make the CLI less buggy
- Parsing = agent + experimental runs doesn't quite match up, just make it a bit better lol

## set sail?!
Refer to "This will lend itself..." for next big step.
ReAct prompting? -- could be good to daisy-chain together data, especially with the structured database, chaining experiments, etc.