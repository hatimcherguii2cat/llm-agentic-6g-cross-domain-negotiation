# LLM-powered Agentic 6G Cross-Domain Negotiation

This project simulates an A2A-like negotiation between two LLM-powered agents, a RAN (Energy Saving) Agent and an Edge (Latency Assurance) Agent, to reconcile their conflicting goals. They use a small-scale digital twin (DT) to validate their proposals/counter-proposals before sending them to the peer agent. The simulation evaluates different strategies, including using a collective memory with and without debiasing mechanisms.

<img width="661" height="505" alt="image" src="https://github.com/user-attachments/assets/c72f0168-9a94-40b9-8cf7-37a367c590c6" />



## Citation 
If you use this code or any (modified) part of it, please cite it as: 
```bibtex
@ARTICLE{11517469,
  author={Chergui, Hatim and Rezazadeh, Farhad and Debbah, Merouane and Verikoukis, Christos},
  journal={IEEE Open Journal of the Communications Society}, 
  title={A Tutorial on Cognitive Biases in Agentic AI-Driven 6G Autonomous Networks}, 
  year={2026},
  volume={7},
  number={},
  pages={5214-5240},
  keywords={Memory;Modeling;Cognition;Cognitive systems;Radio access networks;Regional area networks;Large language models;Planning;Tools;Proposals;6G;agentic AI;bias;network automation},
  doi={10.1109/OJCOMS.2026.3692946}}

```

## Project Structure

- `main.py`: The main entry point to run the simulation and generate plots.
- `config.py`: Contains global simulation parameters and constants.
- `network_simulator.py`: Defines the core `NetworkSimulator` class, which models the network environment.
- `e2_api_tool.py`: Provides the `E2APISimulator`, an interface for agents to interact with the network simulator.
- `digital_twin.py`: Contains the `DigitalTwin` class, a model used by agents for internal testing of proposals.
- `collective_memory.py`: Implements the `CollectiveMemory` class for storing and retrieving negotiation strategies.
- `llm_agent.py`: Defines the base `LLMAgent` class for the negotiating agents.
- `agents.py`: Contains the specialized `RanAgent` and `EdgeAgent` classes.
- `a2a.py`: Implements the `A2ANegotiationManager` to orchestrate the negotiation process.
- `negotiation_parser.py`: Implements the parsing of negotiation messages.
- `requirements.txt`: Lists the necessary Python packages for this project.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Key:**
    Make sure your `GOOGLE_API_KEY` is set as an environment variable.

3.  **Run the Simulation:**
    ```bash
    python main.py
    ```
    The first run will execute the full simulation and save the results to `simulation_results.pkl`. Subsequent runs will load from this file to generate plots without re-running the simulation. To force a new simulation, delete `simulation_results.pkl`.
