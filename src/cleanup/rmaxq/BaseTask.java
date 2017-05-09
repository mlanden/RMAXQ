package cleanup.rmaxq;

import burlap.mdp.core.action.ActionType;
import rmaxq.framework.PrimitiveTaskNode;

public class BaseTask extends PrimitiveTaskNode {

	public BaseTask(ActionType a){
		this.setActionType(a);
	}
}
