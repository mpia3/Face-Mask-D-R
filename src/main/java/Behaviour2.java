import jade.core.behaviours.Behaviour;

public class Behaviour2 extends Behaviour{
	private int iterazione=0;
	public void action(){
		System.out.println("L'agente " + myAgent.getAID().getName() + " sta eseguendo il behaviour2");
		iterazione++;
	}
	public boolean done(){	return iterazione==2;}
}
