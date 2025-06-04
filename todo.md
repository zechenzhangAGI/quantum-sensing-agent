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

# TODO --03 June
## We have implemented a RAG system in the agent.py that searches past converstation history to find new information when executing new tasks. However, it is done using the user content as embedding search and on the other hand also saves the entire conversation history as one emedding text file. Instead, we want to
--1. Make the agent store the embeddings as chunks of texts every few tunrs (where the number of turns can be set, default is 10 let's say)
--2. When retrieving the texts using RAG, we want to implement a new strategy where the agent has the option to search through past history using RAG when it gets stuck, ie. it can use RAG as a tool. in this case, think about how to implement this. For example, the query embedding should be the recent conversation history rather than the pure user content. Maybe you should also change system prompt since now RAG is a tool for the agent to retrieve relevant info from past. 
--3. Right now it seems like I have not implemented the cached input from anthropic. checkout how to implement that to save money 

  Note: The agent.py file is written for the final environment on windows that controls the actual experiments. You can create a mock test environment which might be the agent_mac.py file (but you have to tweek it along) to test the implemented functionalities. also read the test scripts for reference.  

