package rmaxq.framework;

import java.util.ArrayList; 
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.SolverDerivedPolicy;
import burlap.behavior.singleagent.Episode;
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
	private Map<GroundedTask, List<HashableState>> envolope;
	
	//ta 
	private Map<GroundedTask, List<HashableState>> terminal;
	
	private double dynamicPrgEpsilon;
	private int threshold;
	private TaskNode root;
	private HashableStateFactory hashingFactory;
	private double Vmax;
	private Environment env;
	private State initialState;
	
	private List<HashableState> reachableStates = new ArrayList<HashableState>();
	private long time = 0;
	public RmaxQLearningAgent(TaskNode root, HashableStateFactory hs, double vmax, int threshold, double maxDelta){
		this.root = root;
		this.reward = new HashMap<GroundedTask, Map<HashableState,Double>>();
		this.transition = new HashMap<GroundedTask, Map<HashableState, Map<HashableState, Double>>>();
		this.totalReward = new HashMap<HashableState, Map<GroundedTask,Double>>();
		this.actionCount = new HashMap<HashableState, Map<GroundedTask,Integer>>();
		this.qProvider = new HashMap<GroundedTask, QProviderRmaxQ>();
		this.envolope = new HashMap<GroundedTask, List<HashableState>>();
		this.resultingStateCount = new HashMap<HashableState, Map<GroundedTask,Map<HashableState,Integer>>>();
		this.terminal = new HashMap<GroundedTask, List<HashableState>>();
		this.qPolicy = new HashMap<GroundedTask, SolverDerivedPolicy>();
		this.groundedTaskMap = new HashMap<String, GroundedTask>();
		this.dynamicPrgEpsilon = maxDelta;
		this.hashingFactory = hs;
		this.Vmax = vmax;
		this.threshold = threshold;
	}
	 
	public long getTime(){
		return time;
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
		
		//look at equals in grounded task
		time = System.currentTimeMillis();
		HashableState hs = hashingFactory.hashState(env.currentObservation());
		e = R_MaxQ(hs, rootSolve, e);
//		System.out.println(this.transition.keySet().size());
		time = System.currentTimeMillis() - time;
		return e;
	}

	protected Episode R_MaxQ(HashableState hs, GroundedTask task, Episode e){
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
				computePolicy(hs, task, true);
				
				if(!qProvider.containsKey(task))
					qProvider.put(task, new QProviderRmaxQ(hashingFactory, task));
				QProviderRmaxQ qvalues = qProvider.get(task);
				
				if(!qPolicy.containsKey(task))
					qPolicy.put(task, new GreedyQPolicy());
				SolverDerivedPolicy taskFromPolicy = qPolicy.get(task);
				taskFromPolicy.setSolver(qvalues);
				
				Action maxqAction = taskFromPolicy.action(hs.s());
				if(!groundedTaskMap.containsKey(maxqAction.actionName()))
					addChildTasks(task, hs.s());
				
				//R pia(s') (s')
				GroundedTask childFromPolicy = groundedTaskMap.get(maxqAction.actionName());
				e = R_MaxQ(hs, childFromPolicy , e);
				State s = e.stateSequence.get(e.stateSequence.size() - 1);
				hs = hashingFactory.hashState(s);
				
				terminal = task.t.terminal(s, task.action);
			}while(!terminal);
			
			return e;
			
		}
	}
	
	public void computePolicy(HashableState hs, GroundedTask task, boolean prepare){
		if(prepare)
			prepareEnvolope(hs, task);
		if(!qProvider.containsKey(task))
			qProvider.put(task, new QProviderRmaxQ(hashingFactory, task));
		QProviderRmaxQ qp = qProvider.get(task);
		
		boolean converged = false;
		while(!converged){   
			double maxDelta = 0;
			if(!envolope.containsKey(task))
				envolope.put(task, new ArrayList<HashableState>());
			List<HashableState> envolopA = envolope.get(task);
			
			for(int i = 0; i < envolopA.size(); i++){
				HashableState hsprime = envolopA.get(i);
				List<GroundedTask> ActionIns = getTaskActions(task, hsprime.s());
				for(int j  = 0; j < ActionIns.size(); j++){
					GroundedTask a = ActionIns.get(j);
					
					double oldQ = qp.qValue(hsprime.s(), a.action);
					
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
					qp.update(hsprime.s(), a.action, newQ);
					
					if(Math.abs(oldQ - newQ) > maxDelta)
						maxDelta = Math.abs(oldQ - newQ);
				}
			}
			if(maxDelta < dynamicPrgEpsilon)
				converged = true;
		}
	}
	
	public void prepareEnvolope(HashableState hs, GroundedTask task){
		if(!envolope.containsKey(task))
			envolope.put(task, new ArrayList<HashableState>());
		List<HashableState> envelope = envolope.get(task);
		
		if(!envelope.contains(hs)){
			envelope.add(hs);
			List<GroundedTask> ActionIns = getTaskActions(task, hs.s());
			for(int i = 0; i < ActionIns.size(); i++){
				GroundedTask a = ActionIns.get(i);
				computeModel(hs, a); 
				
				//get function forPa'(s, .)
				if(!transition.containsKey(a))
					transition.put(a, new HashMap<HashableState, Map<HashableState,Double>>());
				if(!transition.get(a).containsKey(hs)){
					transition.get(a).put(hs, new HashMap<HashableState, Double>());
				}
				
				Map<HashableState, Double> psa = transition.get(a).get(hs);
				for(HashableState hsp : psa.keySet()){
					if(psa.get(hsp) > 0)
						prepareEnvolope(hsp, task);
				}
			}
		}
	}
	
	public void computeModel(HashableState hs, GroundedTask task){
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
			computePolicy(hs, task, true);
			
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
				//temporary holders for batch updates
				Map <HashableState, Map<HashableState, Double>> tempTransition 
					= new HashMap<HashableState, Map<HashableState,Double>>();
	
				if(!envolope.containsKey(task))
					envolope.put(task, new ArrayList<HashableState>());
				List<HashableState> envelopeA = envolope.get(task);
				for(HashableState hsprime : envelopeA){
					//for all s in ta
					// equation 7
					List<HashableState> terminal = getTerminalStates(task);
					for(HashableState hx :terminal){
						//get current pa(s',x)
						if(!transition.containsKey(task))
							transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
						if(!transition.get(task).containsKey(hsprime))
							transition.get(task).put(hsprime, new HashMap<HashableState, Double>());
						if(!transition.get(task).get(hsprime).containsKey(hx))
							transition.get(task).get(hsprime).put(hx, 0.);
						double oldPrabability = transition.get(task).get(hsprime).get(hx);
					
						Action maxqAction = taskFromPolicy.action(hsprime.s());
						if(!groundedTaskMap.containsKey(maxqAction.actionName()))
							addChildTasks(task, hsprime.s());
						
						GroundedTask childFromPolicy = groundedTaskMap.get(maxqAction.actionName());
						
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
						// eq 5
						for(HashableState hnext: childFromSp.keySet()){
							if(task.t.terminal(hnext.s(), task.action))
								continue;
							
							double psprimeTospprime = childFromSp.get(hnext);
							//pa (s'',x)
							if(!transition.containsKey(task))
								transition.put(task, new HashMap<HashableState, Map<HashableState,Double>>());
							if(!transition.get(task).containsKey(hnext))
								transition.get(task).put(hnext, new HashMap<HashableState, Double>());
							if(!transition.get(task).get(hnext).containsKey(hx))
								transition.get(task).get(hnext).put(hx, 0.);
							double pspptohx = transition.get(task).get(hnext).get(hx);
							
							weightedTransition += psprimeTospprime * pspptohx;
						}
						double newProb = childProbability + weightedTransition;
						
						if(Math.abs(newProb - oldPrabability) > maxChange)
							maxChange = Math.abs(newProb - oldPrabability);
						
						//set pa(s',x)
						if(!tempTransition.containsKey(hsprime))
							tempTransition.put(hsprime, new HashMap<HashableState, Double>());
						tempTransition.get(hsprime).put(hx, newProb);
					}
//					System.out.println(maxChange);
					if(maxChange < dynamicPrgEpsilon)
						converged = true;
					
					//overwrite reward and transition
					
					for(HashableState s1 : tempTransition.keySet()){
						for(HashableState s2 : tempTransition.get(s1).keySet()){
							this.transition.get(task).get(s1).put(s2, tempTransition.get(s1).get(s2));
						}
					}
				}
			}		
				
					
			converged = false;
			while(!converged){
				double maxChange = 0;
				
				//temporary holders for batch updates
				Map<HashableState, Double> tempReward = new HashMap<HashableState, Double>();
	
				if(!envolope.containsKey(task))
					envolope.put(task, new ArrayList<HashableState>());
				List<HashableState> envelopeA = envolope.get(task);
				for(HashableState hsprime : envelopeA){
					
					if(!reward.containsKey(task))
						reward.put(task, new HashMap<HashableState, Double>());
					if(!reward.get(task).containsKey(hsprime))
						reward.get(task).put(hsprime, Vmax);
					double prevReward = reward.get(task).get(hsprime);
//					double oldValue = qvalues.value(sprime);

					Action maxqAction = taskFromPolicy.action(hsprime.s());
					if(!groundedTaskMap.containsKey(maxqAction.actionName()))
						addChildTasks(task, hsprime.s());
					
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
					// equation 4
					for(HashableState hnext : childProbabilities.keySet()){
						//get Ra(nextstate)
						if(task.t.terminal(hnext.s(), task.action))
							continue;
						
						if(!reward.containsKey(task))
							reward.put(task, new HashMap<HashableState, Double>());
						if(!reward.get(task).containsKey(hnext))
							reward.get(task).put(hnext, Vmax);
						double nextReward = reward.get(task).get(hnext);
						
						weightedReward += childProbabilities.get(hnext) * nextReward;
					}
					
					if(!reward.containsKey(task))
						reward.put(task, new HashMap<HashableState, Double>());
					
					double newReward = actionReward + weightedReward;
					
					tempReward.put(hsprime, newReward);
					
					//find max change for value iteration
					if(Math.abs(newReward - prevReward) > maxChange)
						maxChange = Math.abs(newReward - prevReward);
					
					
					
				}
//				System.out.println(maxChange);
				if(maxChange < dynamicPrgEpsilon)
					converged = true;
				
				//overwrite reward and transition
				for(HashableState hashState : tempReward.keySet()){
					this.reward.get(task).put(hashState, tempReward.get(hashState));
				}
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
	
	protected List<HashableState> getTerminalStates(GroundedTask t){
		if(terminal.containsKey(t))
			return terminal.get(t);
  		List<HashableState> terminals = new ArrayList<HashableState>();
		for(HashableState s :reachableStates){
			if(t.t.terminal(s.s(), t.getAction()))
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
