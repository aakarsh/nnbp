#ifndef _PY_INTERFACE_H_
#define _PY_INTERFACE_H_

#include <predictor.h>
#include <Python.h>
/***********************************************************/

#ifdef __cplusplus
extern "C" {
#endif


/* Header file for perceptron interface */


static PyObject *
spam_system(PyObject *self, PyObject *args)
{
  const char *command;
  int sts;

  if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;
  sts = system(command);
  return Py_BuildValue("i", sts);
}



PyMODINIT_FUNC
initspam(void)
{
  PyObject *m;

  m = Py_InitModule("spam", SpamMethods);
  if (m == NULL)
    return;

  SpamError = PyErr_NewException("spam.error", NULL, NULL);
  Py_INCREF(SpamError);
  PyModule_AddObject(m, "error", SpamError);
}


static PyObject *
spam_system(PyObject *self, PyObject *args)
{
  const char *command;
  int sts;

  if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;
  sts = system(command);
  if (sts < 0) {
    PyErr_SetString(SpamError, "System command failed");
    return NULL;
  }
  return PyLong_FromLong(sts);
}


  
static int
import_spam(void)
{
  //PySpam_API = (void **)PyCapsule_Import("spam._C_API", 0);
  //  return (PySpam_API != NULL) ? 0 : -1;
  return 0;
}


#ifdef __cplusplus
}
#endif
/***********************************************************/
#endif

