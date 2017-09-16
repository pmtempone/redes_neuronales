library(ggplot2)

Clase_A = t(as.matrix(data.frame(c(1, 3) , c(1.5, 4), c(2, 3), c(1.5, 2.5), c(2.5, 3))))

Clase_B = t(as.matrix(data.frame(c(2.5, 4) , c(2.5, 2), c(3, 3), c(3, 2.5), c(1.5, 5), c(2, 4.5))))
  
Clase_A <- cbind(Clase_A,Clase='A')
Clase_B <- cbind(Clase_B,Clase='B')

conjunto <- rbind.data.frame(Clase_A,Clase_B)

conjunto$V1 <- as.numeric(as.character(conjunto$V1))
conjunto$V2 <- as.numeric(as.character(conjunto$V2))

ggplot(conjunto,aes(x=V1,y=V2,col=Clase))+geom_point()+geom_abline(intercept = -2.5, slope = 2)+geom_abline(intercept = 6,slope = -0.9,colour='green')


write.table(conjunto,"salida_multiperceptron.csv",row.names = FALSE,sep = ";",dec = ",")
