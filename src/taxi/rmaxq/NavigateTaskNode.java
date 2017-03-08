package taxi.rmaxq;

import java.util.ArrayList;
import java.util.List;

import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.UniformCostRF;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import rmaxq.framework.GroundedTask;
import rmaxq.framework.NonPrimitiveTaskNode;
import rmaxq.framework.TaskNode;
import taxi.state.TaxiLocation;
import taxi.state.TaxiState;

/**
 * Created by ngopalan on 8/14/16.
 */
public class NavigateTaskNode extends NonPrimitiveTaskNode{


//    OOSADomain l0Domain;

	public static final String ACTION_NAVIGATE = "navigate";
    protected String[] locations;
    protected List<GroundedTask> gts;
    public NavigateTaskNode(String[] locs, TaskNode[] children, OOSADomain dom){
        this.name = ACTION_NAVIGATE;
        this.domain = dom;
        locations = locs;
    	this.taskNodes = children;
    	
    	gts = new ArrayList<GroundedTask>();
    	RewardFunction urf = new UniformCostRF();
    	for( String loc : locations){
        	gts.add(new GroundedTask(this, new SimpleAction(ACTION_NAVIGATE + "_" + loc), urf));
        }
    }

    @Override
    public Object parametersSet(State s) {
        return locations;
    }


    @Override
    public boolean terminal(State s, Action action) {
        String goal = action.actionName().split("_")[1];
        TaxiState st = (TaxiState) s;
        int tx = st.taxi.x, ty = st.taxi.y;
        for(TaxiLocation l : st.locations){
        	if(l.name().equals(goal)){
        		return tx == l.x && ty == l.y;
        	}
        }
        return false;
    }

    @Override
    public List<GroundedTask> getApplicableGroundedTasks(State s) {
        List<GroundedTask> gtList = new ArrayList<GroundedTask>();
        for(GroundedTask gt : gts){
        	if(!terminal(s, gt.getAction())){
        		gtList.add(gt);
        	}
        }
        
        return gtList;
    }
}
