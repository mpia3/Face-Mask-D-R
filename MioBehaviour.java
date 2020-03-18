import jade.core.behaviours.Behaviour;

public class MioBehaviour extends Behaviour{
	public void action(){
		System.out.println("L'agente " + myAgent.getAID().getName() + " sta eseguendo il behaviour1");
		myAgent.addBehaviour(new Behaviour2());
	}
	public boolean done(){	return true;}
}