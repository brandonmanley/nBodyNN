#!/usr/bin/env wolframscript

k=1; 

While[ k<100, 

       k++; Pause@.1; 

      (* progress indicator *) 
       Run[ "echo -n '" <> 
            StringRepeat["|",k] <>  
            StringRepeat["-", 100-k] <> 
            "\r'"]
     ]