package rmaxq.framework;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.SolverDerivedPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import burlap.statehashing.HashableState;
import burlap.statehashing.HashableStateFactory;

public class RmaxQLearningAgent implements LearningAgent {

	//Qa(s', a')
//	private Map<GroundedTask, Map<HashableState, Map<GroundedTask, Double>>> qValue;
		
	//Va(s)
//	private Map<GroundedTask, Map<HashableState, Double>> value;
	
	//Pa(s', x)
	private Map<GroundedTask, Map<HashableState, Map<HashableState, Double>>> transition;
	
	//Ra(s) 
	private Map<GroundedTask, Map<HashableState, Double>> reward;
	//pi a
	
	//r(s,a)
	private Map<HashableState, Map< GroundedTask, Double>> totalReward;
	
	//n(s,a)
	private Map<HashableState, Map< GroundedTask, Integer>> actionCount;
	
	//n(s,a,s')
	private Map<HashableState, Map< GroundedTask, Map<HashableState, Integer>>> resultingStateCount;
	
	//grounded task map
	private Map<String, GroundedTask> groundedTaskMap;
	
	//QProviders for each grounded task
	private Map<GroundedTask, QProviderRmaxQ> qProvider;
	
	//policies
	private Map<GroundedTask, SolverDerivedPolicy> qPolicy;
	
	//envolope(a)
	private Map<GroundedTask, List<State>> envolope;
	
	//ta 
	private Map<GroundedTask, List<State>> terminal;
	
	private double dynamicPrgEpsilon;
	private int threshold;
	private TaskNode root;
	private HashableStateFactory hashingFactory;
	private double Vmax;
	private Environment env;
	private State initialState;
	
	private List<State> reachableStates = new ArrayList<State>();
	private long time = 0;
	public RmaxQLearningAgent(TaskNode root, HashableStateFactory hs, double vmax, int threshold, double maxDelta){
		this.root = root;
		this.reward = new HashMap<GroundedTask, Map<HashableState,Double>>();
		this.transition = new HashMap<GroundedTask, Map<HashableState, Map<HashableState, Double>>>();
		this.totalReward = new HashMap<HashableState, Map<GroundedTask,Double>>();
		this.actionCount = new HashMap<HashableState, Map<GroundedTask,Integer>>();
		this.qProvider = new HashMap<GroundedTask, QProviderRmaxQ>();
		this.envolope = new HashMap<GroundedTask, List<State>>();
		this.resultingStateCount = new HashMap<HashableState, Map<GroundedTask,Map<HashableState,Integer>>>();
		this.terminal = new HashMap<GroundedTask, List<State>>();
		this.qPolicy = new HashMap<GroundedTask, SolverDerivedPolicy>();
		this.groundedTaskMap = new HashMap<String, GroundedTask>();
		this.dynamicPrgEpsilon = maxDelta;
		this.hashingFactory = hs;
		this.Vmax = vmax;
		this.threshold = threshold;
	}
	 
	public Episode runLearningEpisode(Environment env) {
		return runLearningEpisode(env, -1);
	}

	public Episode runLearningEpisode(Environment env, int maxSteps) {
		this.env = env;
		this.initialState = env.currentObservation();
		Episode e = new Episode(initialState);
		GroundedTask rootSolve = root.getApplicableGroundedTasks(env.currentObservation()).get(0);
		reachableStates = StateReachability.getReachableStates(initialState, root.getDomain(), hashingFactory);
		
		try{
			System.out.println(this.transition.get(root).get(initialState).keySet().size());
		}catch(Exception ex){}
		
		time = System.currentTimeMillis();
		e = R_MaxQ(env.currentObservation(), rootSolve, e);
		System.out.println(this.transition.size());
		time = System.currentTimeMillis() - time;
		return e;
	}

	protected Episode R_MaxQ(State s, GroundedTask task, Episode e){
		HashableState hs = hashingFactory.hashState(s);
//		System.out.println(task.actionName());
		
		if(task.t.isTaskPrimitive()){
			Action a = task.getAction();
			EnvironmentOutcome outcome = env.executeAction(a);
			e.transition(outcome);
			State sprime = outcome.op;
 			HashableState hsprime = hashingFactory.hashState(sprime);
			
			//r(s,a) += r
			if(!totalReward.containsKey(hs))
				totalReward.put(hs, new HashMap<GroundedTask, Double>());
			if(!totalReward.get(hs).containsKey(task))
				totalReward.get(hs).put(task, 0.);
			double r = totalReward.get(hs).get(task) + outcome.r;
			totalReward.get(hs).put(task, r);
			
			//n(s,a) ++
			if(!actionCount.containsKey(hs))
				actionCount.put(hs, new HashMap<GroundedTask, Integer>());
			if(!actionCount.get(hs).containsKey(task))
				actionCount.get(hs).put(task, 0);
			int n = actionCount.get(hs).get(task) + 1;
			actionCount.get(hs).put(task, n);
			
			//n(s,a,s')++
			if(!resultingStateCount.containsKey(hs))
				resultingStateCount.put(hs, new HashMap<GroundedTask, Map<HashableState,Integer>>());
			if(!resultingStateCount.get(hs).containsKey(task))
				resultingStateCount.get(hs).put(task, new HashMap<HashableState, Integer>());
			if(!resultingStateCount.get(hs).get(task).containsKey(hsprime))
				resultingStateCount.get(hs).get(task).put(hsprime, 0);
			n = resultingStateCount.get(hs).get(task).get(hsprime) + 1;
			resultingStateCount.get(hs).get(task).put(hsprime, n);
			
			//add pa(s, sprime) =0 in order to preform the update in compute model
			if(!transition.containsKey(task))
				transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
			if(!transition.get(task).containsKey(hs))
				transition.get(task).put(hs, new HashMap<HashableState, Double>());
			if(!transition.get(task).get(hs).containsKey(hsprime))
				transition.get(task).get(hs).put(hsprime, 0.);
			
			return e;
		}else{ //task is composute
			boolean terminal = false;
			do{
				computePolicy(s, task, true);
				
				if(!qProvider.containsKey(task))
					qProvider.put(task, new QProviderRmaxQ(hashingFactory, task));
				QProviderRmaxQ qvalues = qProvider.get(task);
				
				if(!qPolicy.containsKey(task))
					qPolicy.put(task, new GreedyQPolicy());
				SolverDerivedPolicy taskFromPolicy = qPolicy.get(task);
				taskFromPolicy.setSolver(qvalues);
				
				Action maxqAction = taskFromPolicy.action(s);
				if(!groundedTaskMap.containsKey(maxqAction.actionName()))
					addChildTasks(task, s);
				
				//R pia(s') (s')
				GroundedTask childFromPolicy = groundedTaskMap.get(maxqAction.actionName());
				e = R_MaxQ(s, childFromPolicy , e);
				s = e.stateSequence.get(e.stateSequence.size() - 1);
				
				terminal = task.t.terminal(s, task.action);
			}while(!terminal);
			
			return e;
			
		}
	}
	
	public void computePolicy(State s, GroundedTask task, boolean prepare){
		if(prepare)
			prepareEnvolope(s, task);
		if(!qProvider.containsKey(task))
			qProvider.put(task, new QProviderRmaxQ(hashingFactory, task));
		QProviderRmaxQ qp = qProvider.get(task);
		
		boolean converged = false;
		while(!converged){   
			double maxDelta = 0;
			if(!envolope.containsKey(task))
				envolope.put(task, new ArrayList<State>());
			List<State> envolopA = envolope.get(task);
			
			for(int i = 0; i < envolopA.size(); i++){
				State sprime = envolopA.get(i);
				HashableState hsprime = hashingFactory.hashState(sprime);
				List<GroundedTask> ActionIns = getTaskActions(task, sprime);
				for(int j  = 0; j < ActionIns.size(); j++){
					GroundedTask a = ActionIns.get(j);
					
					double oldQ = qp.qValue(sprime, a.action);
					
					//Ra'(s')
					if(!reward.containsKey(a))
						reward.put(a, new HashMap<HashableState, Double>());
					if(!reward.get(a).containsKey(hsprime))
						reward.get(a).put(hsprime, Vmax);
					double actionReward = reward.get(a).get(hsprime);
					
					if(!transition.containsKey(a))
						transition.put(a, new HashMap<HashableState, Map<HashableState,Double>>());
					if(!transition.get(a).containsKey(hsprime))
						transition.get(a).put(hsprime, new HashMap<HashableState, Double>());
					Map<HashableState, Double> fromsp = transition.get(a).get(hsprime);
					
					double weightedQ = 0;
					for(HashableState hspprime : fromsp.keySet()){
						double value = qp.value(hspprime.s());
						weightedQ += fromsp.get(hspprime) * value;
					}
					double newQ = actionReward + weightedQ;
					qp.update(sprime, a.action, newQ);
					
					if(Math.abs(oldQ - newQ) > maxDelta)
						maxDelta = Math.abs(oldQ - newQ);
				}
			}
			if(maxDelta < dynamicPrgEpsilon)
				converged = true;
		}
	}
	
	public void prepareEnvolope(State s, GroundedTask task){
		if(!envolope.containsKey(task))
			envolope.put(task, new ArrayList<State>());
		List<State> envelope = envolope.get(task);
		HashableState hs = hashingFactory.hashState(s);
		
		if(!envelope.contains(s)){
			envelope.add(s);
			List<GroundedTask> ActionIns = getTaskActions(task, s);
			for(int i = 0; i < ActionIns.size(); i++){
				GroundedTask a = ActionIns.get(i);
				computeModel(s, a); 
				
				//get function forPa'(s, .)
				if(!transition.containsKey(a))
					transition.put(a, new HashMap<HashableState, Map<HashableState,Double>>());
				if(!transition.get(a).containsKey(hs)){
					transition.get(a).put(hs, new HashMap<HashableState, Double>());
				}
				
				Map<HashableState, Double> psa = transition.get(a).get(hs);
				for(HashableState hsp : psa.keySet()){
					if(psa.get(hsp) > 0)
						prepareEnvolope(hsp.s(), task);
				}
			}
		}
	}
	
	public void computeModel(State s, GroundedTask task){
		HashableState hs = hashingFactory.hashState(s);
		if(task.t.isTaskPrimitive()){
			//n(s, a)
			if(!actionCount.containsKey(hs))
				actionCount.put(hs, new HashMap<GroundedTask, Integer>());
			if(!actionCount.get(hs).containsKey(task))
				actionCount.get(hs).put(task, 0);
			int n_sa = actionCount.get(hs).get(task);
			
			if(n_sa >= threshold){
				//r(s, a)
				if(!totalReward.containsKey(hs))
					totalReward.put(hs, new HashMap<GroundedTask, Double>());
				if(!totalReward.get(hs).containsKey(task))
					totalReward.get(hs).put(task, 0.);
				double r_sa = totalReward.get(hs).get(task);
				
				//set Ra(s) to r(s,a) / n(s,a)
				if(!reward.containsKey(task))
					reward.put(task, new HashMap<HashableState, Double>());
				
				double newR = r_sa / n_sa;
				reward.get(task).put(hs, newR);
				
				//get Pa(s, .)
				if(!transition.containsKey(task))
					transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
				if(!transition.get(task).containsKey(hs))
					transition.get(task).put(hs, new HashMap<HashableState, Double>());
				
				for(HashableState hsprime : transition.get(task).get(hs).keySet()){
					//get n(s, a, s')
					if(!resultingStateCount.containsKey(hs))
						resultingStateCount.put(hs, new HashMap<GroundedTask, Map<HashableState,Integer>>());
					if(!resultingStateCount.get(hs).containsKey(task))
						resultingStateCount.get(hs).put(task, new HashMap<HashableState, Integer>());
					if(!resultingStateCount.get(hs).get(task).containsKey(hsprime))
						resultingStateCount.get(hs).get(task).put(hsprime, 0);
					int n_sasp = resultingStateCount.get(hs).get(task).get(hsprime);
					
					//set Pa(s, s') = n(s,a,s') / n(s, a)
					double p_assp = n_sasp / n_sa;
					if(!transition.containsKey(task))
						transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
					if(!transition.get(task).containsKey(hs))
						transition.get(task).put(hs, new HashMap<HashableState, Double>());
					transition.get(task).get(hs).put(hsprime, p_assp);
				}
			}
		}else{
			computePolicy(s, task, true);
			
			if(!qProvider.containsKey(task))
				qProvider.put(task, new QProviderRmaxQ(hashingFactory, task));
			QProviderRmaxQ qvalues = qProvider.get(task);
			
			if(!qPolicy.containsKey(task))
				qPolicy.put(task, new GreedyQPolicy());
			SolverDerivedPolicy taskFromPolicy = qPolicy.get(task);
			taskFromPolicy.setSolver(qvalues);
			
			boolean converged = false;
			while(!converged){
				double maxChange = 0;
				
				if(!envolope.containsKey(task))
					envolope.put(task, new ArrayList<State>());
				List<State> envelopeA = envolope.get(task);
				for(State sprime : envelopeA){
					 
					HashableState hsprime = hashingFactory.hashState(sprime);
					double oldValue = qvalues.value(sprime);

					Action maxqAction = taskFromPolicy.action(sprime);
					if(!groundedTaskMap.containsKey(maxqAction.actionName()))
						addChildTasks(task, sprime);
					
					//R pia(s') (s')
					GroundedTask childFromPolicy = groundedTaskMap.get(maxqAction.actionName());
					if(!reward.containsKey(childFromPolicy))
						reward.put(childFromPolicy, new HashMap<HashableState, Double>());
					if(!reward.get(childFromPolicy).containsKey(hsprime))
						reward.get(childFromPolicy).put(hsprime, Vmax);
					double actionReward = reward.get(childFromPolicy).get(hsprime);
					
					//p pia(s') (s',.) 
					if(!transition.containsKey(childFromPolicy))
						transition.put(childFromPolicy, new HashMap<HashableState, Map<HashableState,Double>>());
					if(!transition.get(childFromPolicy).containsKey(hsprime))
						transition.get(childFromPolicy).put(hsprime, new HashMap<HashableState, Double>());
					Map<HashableState, Double> childProbabilities = 
							this.transition.get(childFromPolicy).get(hsprime);
					
					double weightedReward = 0;
					for(HashableState nextState : childProbabilities.keySet()){
						//get Ra(nextstate)
						if(task.t.terminal(nextState.s(), task.action))
							continue;
						
						HashableState hnext = hashingFactory.hashState(nextState.s());
						if(!reward.containsKey(task))
							reward.put(task, new HashMap<HashableState, Double>());
						if(!reward.get(task).containsKey(hnext))
							reward.get(task).put(hnext, Vmax);
						double nextReward = reward.get(task).get(hnext);
						
						weightedReward += childProbabilities.get(nextState) * nextReward;
					}
					
					if(!reward.containsKey(task))
						reward.put(task, new HashMap<HashableState, Double>());
					
					double newReward = actionReward + weightedReward;
					reward.get(task).put(hsprime, newReward);
					
					//find max change for value iteration
//					if(Math.abs(newReward - prevReward) > maxChange)
//						maxChange = Math.abs(newReward - prevReward);
					
					//for all s in ta
					List<State> terminal = getTerminalStates(task);
					for(State x :terminal){
						HashableState hx = hashingFactory.hashState(x);
						//get current pa(s',x)
//						if(!transition.containsKey(task))
//							transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
//						if(!transition.get(task).containsKey(hsprime))
//							transition.get(task).put(hsprime, new HashMap<HashableState, Double>());
//						if(!transition.get(task).get(hsprime).containsKey(hx))
//							transition.get(task).get(hsprime).put(hx, 0.);
//						double oldPrabability = transition.get(task).get(hsprime).get(hx);
						
						//p pia(s) (s',x)
						if(!transition.containsKey(childFromPolicy))
							transition.put(childFromPolicy, new HashMap<HashableState, Map<HashableState,Double>>());
						if(!transition.get(childFromPolicy).containsKey(hsprime))
							transition.get(childFromPolicy).put(hsprime, new HashMap<HashableState, Double>());
						if(!transition.get(childFromPolicy).get(hsprime).containsKey(hx))
							transition.get(childFromPolicy).get(hsprime).put(hx, 0.);
						double childProbability = transition.get(childFromPolicy).get(hsprime).get(hx);
						
						double weightedTransition = 0;
						Map<HashableState, Double> childFromSp = transition.get(childFromPolicy).get(hsprime);
						//sum over all p pia(s) (s',.)
						for(HashableState hspprime: childFromSp.keySet()){
							if(task.t.terminal(hspprime.s(), task.action))
								continue;
							
							double psprimeTospprime = childFromSp.get(hspprime);
							//pa (s'',x)
							if(!transition.containsKey(task))
								transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
							if(!transition.get(task).containsKey(hspprime))
								transition.get(task).put(hspprime, new HashMap<HashableState, Double>());
							if(!transition.get(task).get(hspprime).containsKey(hx))
								transition.get(task).get(hspprime).put(hx, 0.);
							double pspptozx = transition.get(task).get(hspprime).get(hx);
							
							weightedTransition += psprimeTospprime * pspptozx;
						}
						double newProb = childProbability + weightedTransition;
						
//						if(Math.abs(newProb - oldPrabability) > maxChange)
//							maxChange = Math.abs(newProb - oldPrabability);
						
						//set pa(s',x)
						if(!transition.containsKey(task))
							transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
						if(!transition.get(task).containsKey(hsprime))
							transition.get(task).put(hsprime, new HashMap<HashableState, Double>());
						transition.get(task).get(hsprime).put(hx, newProb);
					}
					
					//recompute policy and check v
					computePolicy(s, task, true);
					double newValue = qvalues.value(sprime);
					if(Math.abs(newValue - oldValue) > maxChange)
						maxChange = Math.abs(newValue - oldValue);
				}
//				System.out.println(maxChange);
				if(maxChange < dynamicPrgEpsilon)
					converged = true;
			}
		}
	}
		
	protected void addChildTasks(GroundedTask task, State s){
		if(!task.t.isTaskPrimitive()){
			List<GroundedTask> childGroundedTasks =  getTaskActions(task, s);
			
			for(GroundedTask gt : childGroundedTasks){
				if(!groundedTaskMap.containsKey(gt.action.actionName()))
					groundedTaskMap.put(gt.action.actionName(), gt);
			}
		}
	}
	
	protected List<State> getTerminalStates(GroundedTask t){
		if(terminal.containsKey(t))
			return terminal.get(t);
  		List<State> terminals = new ArrayList<State>();
		for(State s :reachableStates){
			if(t.t.terminal(s, t.getAction()))
				terminals.add(s);
		}

		terminal.put(t, terminals);
//		System.out.println(terminals.size()+ " " + t.actionName());
		if(terminals.size() == 0)
			throw new RuntimeException("no terminal");
		return terminals;
	}
	
	public List<GroundedTask> getTaskActions(GroundedTask task, State s){
		TaskNode[] children = task.t.getChildren();
		List<GroundedTask> childTasks = new ArrayList<GroundedTask>();
		for(TaskNode t: children){
			childTasks.addAll(t.getApplicableGroundedTasks(s));
		}
		return childTasks;
	}
}
