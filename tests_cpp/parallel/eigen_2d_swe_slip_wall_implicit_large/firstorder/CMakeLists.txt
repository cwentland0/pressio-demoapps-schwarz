
set(testname eigen_2d_swe_slip_wall_firstorder_implicit_large)
set(exename  ${testname}_exe)

configure_file(../plot.py plot.py COPYONLY)
configure_file(../compare.py compare.py COPYONLY)
configure_file(h_gold.txt h_gold.txt COPYONLY)
configure_file(solution_full_gold.txt solution_full_gold.txt COPYONLY)

add_executable(${exename} ${CMAKE_CURRENT_SOURCE_DIR}/../main.cc)

add_test(NAME ${testname}
COMMAND ${CMAKE_COMMAND}
-DMESHDRIVER=${MESHSRC}/create_full_mesh.py
-DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}
-DEXENAME=$<TARGET_FILE:${exename}>
-DSTENCILVAL=3
-P ${CMAKE_CURRENT_SOURCE_DIR}/../test.cmake
)
