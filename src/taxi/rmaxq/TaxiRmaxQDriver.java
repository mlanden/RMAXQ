package taxi.rmaxq;

import java.util.ArrayList;
import java.util.List;

import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import rmaxq.framework.RmaxQLearningAgent;
import rmaxq.framework.TaskNode;
import taxi.TaxiDomain;
import taxi.TaxiRewardFunction;
import taxi.TaxiTerminationFunction;
import taxi.TaxiVisualizer;
import taxi.state.TaxiLocation;
import taxi.state.TaxiPassenger;
import taxi.state.TaxiState;

public class TaxiRmaxQDriver {
	
	private static SimulatedEnvironment env;
	private static OOSADomain domain;
	
	public static TaskNode setupHeirarcy(){
        TerminalFunction taxiTF = new TaxiTerminationFunction();
        RewardFunction taxiRF = new TaxiRewardFunction(1,taxiTF);

        TaxiDomain TDGen = new TaxiDomain(taxiRF, taxiTF);
        
//        TDGen.setTransitionDynamicsLikeFickleTaxiProlem();
        TDGen.setFickleTaxi(false);
        TDGen.setIncludeFuel(false);
        OOSADomain td = TDGen.generateDomain();
        domain = td;
        State s = TaxiDomain.getSmallClassicState(domain, false);
        env = new SimulatedEnvironment(td, s);

        VisualActionObserver obs = new VisualActionObserver(td, TaxiVisualizer.getVisualizer(5, 5));
        obs.initGUI();
        env.addObservers(obs);
        
        List<TaxiPassenger> passengers = ((TaxiState)s).passengers;
        List<TaxiLocation> locations = ((TaxiState)s).locations;
        String[] locationNames = new String[locations.size()];
        String[] passengerNames = new String[passengers.size()];
        int i = 0;
        for(TaxiPassenger pass : passengers){
        	passengerNames[i] = pass.name();
        	i++;
        }
        i = 0;
        for(TaxiLocation loc : locations){
        	locationNames[i] = loc.name();
        	i++;
        }
        
        
        ActionType east = td.getAction(TaxiDomain.ACTION_EAST);
        ActionType west = td.getAction(TaxiDomain.ACTION_WEST);
        ActionType south = td.getAction(TaxiDomain.ACTION_SOUTH);
        ActionType north = td.getAction(TaxiDomain.ACTION_NORTH);
        ActionType pickup = td.getAction(TaxiDomain.ACTION_PICKUP);
        ActionType dropoff = td.getAction(TaxiDomain.ACTION_DROPOFF);
        
        TaskNode te = new MoveTaskNode(east);
        TaskNode tw = new MoveTaskNode(west);
        TaskNode ts = new MoveTaskNode(south);
        TaskNode tn = new MoveTaskNode(north);
        TaskNode tp = new PickupTaskNode(pickup);
        TaskNode tdp = new PutDownTaskNode(dropoff);
        
        TaskNode[] navigateSubTasks = new TaskNode[]{te, tw, ts, tn};


        TaskNode navigate = new NavigateTaskNode("navigate", locationNames, navigateSubTasks);
        TaskNode[] getNodeSubTasks = new TaskNode[]{tp,navigate};
        TaskNode[] putNodeSubTasks = new TaskNode[]{tdp,navigate};
        
        TaskNode getNode = new GetTaskNode(td, passengerNames, getNodeSubTasks);
        TaskNode putNode = new PutTaskNode(passengerNames, locationNames, putNodeSubTasks);
        
        TaskNode[] rootTasks = new TaskNode[]{getNode, putNode};
        
        TaskNode root = new RootTaskNode("root", td, rootTasks, passengerNames.length );
        
        return root;
	}
	
	public static void runTests(){
	
		LearningAgentFactory RmaxQ = new LearningAgentFactory() {
			
			public String getAgentName() {
				return "R-maxQ";
			}
			
			@Override
			public LearningAgent generateAgent() {
				TaskNode root = setupHeirarcy();
				HashableStateFactory hs = new SimpleHashableStateFactory();
				return new RmaxQLearningAgent(root, hs, 100, 5, 0.01);
			}
		};
		
		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, 1, 100, RmaxQ);
		exp.setUpPlottingConfiguration(1000, 500, 2, 1000,
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.CUMULATIVE_REWARD_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD);

		exp.startExperiment();
	}

	public static void main(String[] args) {
		TaskNode root = setupHeirarcy();
		HashableStateFactory hs = new SimpleHashableStateFactory();
		
//		QlearningState();
		RmaxQLearningAgent RmaxQ = new RmaxQLearningAgent(root, hs, 100, 5, 0.01);

		for(int i = 1; i <= 10; i++){
			Episode e = RmaxQ.runLearningEpisode(env);
			e.write("output/episode_" + i);
			System.out.println("Episode " + i + " time " + RmaxQ.getTime());
			env.resetEnvironment();
		}
//		
//		runTests();
		Visualizer v = TaxiVisualizer.getVisualizer(5, 5);
		EpisodeSequenceVisualizer ep= new EpisodeSequenceVisualizer(v, domain, "output/" );
		ep.setDefaultCloseOperation(ep.EXIT_ON_CLOSE);

//		runTests();
	}

}
