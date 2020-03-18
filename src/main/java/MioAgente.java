import jade.core.Agent;

public class MioAgente extends Agent{
	private String tipoDiBehaviour;
	
	protected void setup() {
		Object[] args=getArguments();
		System.out.println("L'agente " + getAID().getName() + " inizia");
		if (args!=null && args.length>0)
			if (args[0].equals("inizia_da_1"))
					addBehaviour(new MioBehaviour());
			else
					addBehaviour(new Behaviour2());
		else{
			System.out.println("Non hai inserito parametri di ingresso!");
			doDelete();
		}
	}
	protected void takeDown() {
		System.out.println("L'agente " + getAID().getName() + " termina");
	}
}
