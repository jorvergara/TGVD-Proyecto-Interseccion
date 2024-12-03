# -*- coding: utf-8 -*-
"""
Created on Mon May 28 07:12:54 2018

@author: felipe
"""

import vtk;

class Visualization(object):
    """
    Clase Madre Visualization...
    """
    # Constructor de la clase
    def __init__(self, points):
        # We'll create the building blocks of polydata including data attributes.
        self._myPolygon = vtk.vtkPolyData();
        self._myPoints  = vtk.vtkPoints();
        self._myPolys   = vtk.vtkCellArray();
        self._myScalars = vtk.vtkFloatArray();
        self._myPolygonActor = vtk.vtkActor();
        
class Polygon(Visualization):
    """
    Clase Polygon heredada de Visualization...
    """
    def __init__(self, points, faces):
        def mkVtkIdList(it):
            vil = vtk.vtkIdList();
            for i in it:
                vil.InsertNextId(int(i));
            return vil;
        
        # Llamo al constructor de Visualization
        Visualization.__init__(self, points);
        # Load the point, cell, and data attributes.
        for i in range(len(points)):
            self._myPoints.InsertPoint(i, points[i]);
            self._myScalars.InsertTuple1(i,i);
        for i in range(len(faces)):
            self._myPolys.InsertNextCell( mkVtkIdList(faces[i]) );
            
        # We now assign the pieces to the vtkPolyData.
        self._myPolygon.SetPoints(self._myPoints);
        self._myPolygon.SetPolys(self._myPolys);
        
        # Now we'll look at it.
        myPolygonMapper = vtk.vtkPolyDataMapper();
        if vtk.VTK_MAJOR_VERSION <= 5:
            myPolygonMapper.SetInput(self._myPolygon);
        else:
            myPolygonMapper.SetInputData(self._myPolygon);
        myPolygonMapper.SetScalarRange(0,len(points)-1);
        self._myPolygonActor.SetMapper(myPolygonMapper);
        
    def setOpacity(self, opacity):
        self._myPolygonActor.GetProperty().SetOpacity(opacity);
        
    def setColor(self, color):
        self._myPolygonActor.GetProperty().SetColor(color);
    
    def setJetColormap(self):
        self._myPolygon.GetPointData().SetScalars(self._myScalars);
        
class Line(Visualization):
    """
    Clase Line heredada de Visualization...
    """
    def __init__(self, points):
        myLines = [];
        # Llamo al constructor de Visualization
        Visualization.__init__(self, points);
        # Load the point, cell, and data attributes.
        for i in range(len(points)):
            self._myPoints.InsertNextPoint(points[i]);
            self._myScalars.InsertTuple1(i,i);
        
        for i in range(len(points)-1):
            myLines.append(vtk.vtkLine());
            myLines[i].GetPointIds().SetId(0,i);
            myLines[i].GetPointIds().SetId(1,i+1);
            self._myPolys.InsertNextCell(myLines[i]);
        
        # We now assign the pieces to the vtkPolyData.
        self._myPolygon.SetPoints(self._myPoints);
        self._myPolygon.SetLines(self._myPolys);
        
        # Now we'll look at it.
        myPolygonMapper = vtk.vtkPolyDataMapper();
        if vtk.VTK_MAJOR_VERSION <= 5:
            myPolygonMapper.SetInput(self._myPolygon);
        else:
            myPolygonMapper.SetInputData(self._myPolygon);
        myPolygonMapper.SetScalarRange(0,len(points)-1);
        self._myPolygonActor.SetMapper(myPolygonMapper);
        
        self._myPolygonActor.GetProperty().SetLineWidth(1.9);
        
    def setOpacity(self, opacity):
        self._myPolygonActor.GetProperty().SetOpacity(opacity);
    
    def setColor(self, color):
        self._myPolygonActor.GetProperty().SetColor(color);
        
    def setJetColormap(self):
        self._myPolygon.GetPointData().SetScalars(self._myScalars);
        
class Points(Visualization):
    """
    Clase Points heredada de Visualization...
    """
    def __init__(self, listPoints):
        # Llamo al constructor de Visualization
        Visualization.__init__(self, listPoints);
        # Load the point, cell, and data attributes.
        self._myPolys.InsertNextCell(len(listPoints));
        for point in listPoints:
            ID = self._myPoints.InsertNextPoint(point);
            self._myPolys.InsertCellPoint(ID);
            
        # We now assign the pieces to the vtkPolyData.
        self._myPolygon.SetPoints(self._myPoints);
        self._myPolygon.SetVerts(self._myPolys);
        
        # Now we'll look at it.
        myPolygonMapper = vtk.vtkPolyDataMapper();
        if vtk.VTK_MAJOR_VERSION <= 5:
            myPolygonMapper.SetInput(self._myPolygon);
        else:
            myPolygonMapper.SetInputData(self._myPolygon);
        self._myPolygonActor.SetMapper(myPolygonMapper);
        
        self._myPolygonActor.GetProperty().SetPointSize(5);
        
    def setColor(self, color):
        self._myPolygonActor.GetProperty().SetColor(color);
        
    def setPointSize(self, pointSize):
        self._myPolygonActor.GetProperty().SetPointSize(pointSize);
    
class Render(object):
    def __init__(self):
        # The usual rendering stuff.        
        self._myCamera = vtk.vtkCamera();
        self._myCamera.SetPosition(1,1,1);
        self._myCamera.SetFocalPoint(0,0,0);
         
        self._myRenderer = vtk.vtkRenderer();
        self._myRenWin   = vtk.vtkRenderWindow();
        self._myRenWin.AddRenderer(self._myRenderer);
         
        self._myIren = vtk.vtkRenderWindowInteractor();
        self._myIren.SetRenderWindow(self._myRenWin);
        self._myRenWin.SetSize(1440,900);
        
    def SetAxis(self):
        transform = vtk.vtkTransform();
        transform.Translate(0.0, 0.0, 0.0); # axis situado en el origen
        transform.Scale(20.0, 20.0, 20.0); # escala del axis
        
        #========== Genera un axis en el origen ==========
        axes = vtk.vtkAxesActor();
        axes.SetUserTransform(transform);
        
        #========== Etiqueta cada eje ====================
        axes.SetXAxisLabelText("X");
        axes.SetYAxisLabelText("Y");
        axes.SetZAxisLabelText("Z");
        
        #========== Ajusta el ancho del axis =============
        axes.GetXAxisShaftProperty().SetLineWidth(1);
        axes.GetYAxisShaftProperty().SetLineWidth(1);
        axes.GetZAxisShaftProperty().SetLineWidth(1);
        
        #========== Ajusta el tamaño de la fuente ================
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(30);
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(30);
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(30);
        
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone();
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone();
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone();
        
        self._myRenderer.AddActor(axes);
            
    def AddActor(self, polygon):
        self._myRenderer.AddActor(polygon._myPolygonActor);
        self._myRenderer.SetActiveCamera(self._myCamera);
        self._myRenderer.ResetCamera();
        self._myRenderer.SetBackground(1.,1.,1.);
        
    def Start(self):
        self._myRenWin.Render();
        self._myIren.Start();
        
    def setBackground(self, background):
        self._myRenderer.SetBackground(background);
        
    def setWindowSize(self, width, height):
        self._myRenWin.SetSize(width, height);
