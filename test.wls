#!/usr/bin/env wolframscript

k=0; 

While[ k<100, 

       Pause@.1; 
       k++; 

      (* progress indicator *) 
       WriteString[$Output, StringRepeat["|",k] <>  
            StringRepeat["-", 101-k] <> 
            "\r'"]; 

            
     ]