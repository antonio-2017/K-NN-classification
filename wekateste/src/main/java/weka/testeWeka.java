package weka;


 
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class testeWeka {
	
	public static void main (String[] args) throws Exception {
		
        //importação da base de dados de treinamento
		
         DataSource source = new DataSource("treinar.csv");
         Instances Data = source.getDataSet();
         
         
         //espeficicação do atributo classe
         
         Data.setClassIndex(Data.numAttributes() - 1);
         
        
        //Construção do modelo classificador (treinamento)
         
         IBk k3 = new IBk(3);
         k3.buildClassifier(Data);
         
  
        //criação de uma nova instância
         
         /*Instance inst = new Instance(5);
         inst.setDataset(Data);
         inst.setValue(0, "sunny");
         inst.setValue(1, 80);
         inst.setValue(2, 75);
         inst.setValue(3, "TRUE");*/
         
         Instance inst = Data.instance(9);
         
         
         //classificação da nova instância
         
         double classif = k3.classifyInstance(inst);
 
        
         System.out.println(inst.toString());
         
         //imprime o valor classificado
         System.out.println("classificação: " + classif);
         
         //imprime o valor classificado de acordo com o valor do atributo original
         Attribute atributo = Data.attribute(4);
         String classifClass = atributo.value((int) classif);
         System.out.println("classificação: " + classifClass);
         
         
         
       
	}

	

}
