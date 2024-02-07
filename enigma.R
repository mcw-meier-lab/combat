#! /usr/locla/bin/Rscript

library(combat.enigma)

run_enigma_combat <- function(dat, batch, model, eb=TRUE) {
    enigma.harmonized = combat_fit(dat,batch,model,eb)
    enigma.applied = combat_apply(enigma.harmonized,dat,batch,model)
    return(enigma.applied)
}

