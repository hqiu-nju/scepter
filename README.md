# SCEPTer
Simulating Constellation Emission Patterns for Telescopes

This is a repository to simulate the satellite constellation emissions and measure the EPFD of the observed sky area.

The simulation can be performed using a simulated constellation or from real satellite constellation two line elements (TLEs).
Satellite TLEs can be found on https://celestrak.org/

We use the PyCRAF and cysgp4 packages for many of the base calculations, see requirements.txt for list of dependancies, code was written and tested in Python 3.10

