___

# TODO -- 03 Apr 
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

PROBLEM -- should we store default configs and data in the projects dir? (which I'm doing right now)
- This should probably get updated to a contextual call once the database is properly set up.

PROBLEM -- different OS = different filepath convention
- Currently I have agent_mac.py and agent.py, but we should probably somehow detect the OS.
## **SHOULD BE DONE** + all the prompts and whatnot as well

- This will lend itself to being more of a structured database that the agent can knowledge-extract from (e.g. RAG, another action that involves reading from the database structure, etc.)
    - Becomes a dynamic database and long-term memory almost? Especially w/ human supervision
    - No longer starting from scratch W, everything is saved into a log
        - But it has to be more efficient...first glean fundamental rules, then selectively do logs based on similarity to present experiment
    - *improves decision-making*

### b. make the CLI less buggy
- Parsing = agent + experimental runs doesn't quite match up, just make it a bit better lol

## **SHOULD ALSO BE DONE** (not something I can reproduce anymore, @Ruolan please feel free to test as well + send reproducible bugginess)

## set sail?!
Refer to "This will lend itself..." for next big step.
ReAct prompting? -- could be good to daisy-chain together data, especially with the structured database, chaining experiments, etc.