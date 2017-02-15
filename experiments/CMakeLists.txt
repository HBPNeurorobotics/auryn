FILE( GLOB EXP_SOURCES *.cpp )
FOREACH( sourcepath ${EXP_SOURCES} )
	GET_FILENAME_COMPONENT( sourcefile ${sourcepath} NAME )
	STRING( REPLACE ".cpp" "" expname ${sourcefile} )
	ADD_EXECUTABLE( ${expname} ${sourcefile} )
	TARGET_LINK_LIBRARIES( ${expname} auryn ${AURYN_EXT_LINKLIBS} )
ENDFOREACH( sourcepath ${EXP_SOURCES} )

