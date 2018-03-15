import static java.lang.Math.max;
import javax.swing.*;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.awt.*;
import java.awt.image.*;
import java.awt.event.*;
import java.awt.image.IndexColorModel;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import edu.mines.jtk.awt.*;
import edu.mines.jtk.dsp.EigenTensors2;
import edu.mines.jtk.dsp.LocalOrientFilter;
import edu.mines.jtk.io.ArrayInputStream;
import edu.mines.jtk.mosaic.*;
import static edu.mines.jtk.util.ArrayMath.*;
import edu.mines.jtk.util.Quantiler;
import edu.mines.jtk.util.MedianFinder;
import edu.mines.jtk.dsp.*;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;

class ImageFrame extends JFrame {
  public static final int DEFAULT_WIDTH = 800, DEFAULT_HEIGHT = 600;
  public ImageFrame(Image image){
    setTitle("ImageTest");
    setSize(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    ImageComponent component = new ImageComponent(image);
    add(component);
  }
}

class ImageComponent extends JComponent{
  private static final long serialVersionUID = 1L;
  private Image image;
  public ImageComponent(Image image){
    this.image = image;
  }
  public void paintComponent(Graphics g){
    if(image == null) return;
    int imageWidth = image.getWidth(this);
    int imageHeight = image.getHeight(this);
    g.drawImage(image, 0, 0, this);

    /*for (int i = 0; i*imageWidth <= getWidth(); i++)
      for(int j = 0; j*imageHeight <= getHeight();j++)
        if(i+j>0) g.copyArea(0, 0, imageWidth, imageHeight, i*imageWidth, j*imageHeight);*/
  }
}

public class TestJtk {

  public TestJtk(float[][] image) {
    _n1 = image[0].length;
    _n2 = image.length;
    _nv = 1;
    _image = image;
    _st = new StructureTensors(SIGMA,2.0f,1.0f,1.0f,_image);

    int fontSize = 12;
    int width = 720;
    int height = 500;
    int widthColorBar = 60;

    // Plot panel.
    PlotPanel.Orientation ppo = PlotPanel.Orientation.X1DOWN_X2RIGHT;
    PlotPanel.AxesPlacement ppap = PlotPanel.AxesPlacement.LEFT_TOP;
    _panel = new PlotPanel(1,1,ppo,ppap);
    _panel.setColorBarWidthMinimum(widthColorBar);
    _colorBar = _panel.addColorBar();

    // Image view.
    _imageView = _panel.addPixels(_image);
    _imageView.setInterpolation(PixelsView.Interpolation.NEAREST);

    // Tensors view, if visible, on top of paint view.
    float[][][] x12 = getTensorEllipses(_n1,_n2,5,_st);
    float[][] x1 = x12[0];
    float[][] x2 = x12[1];
    _tensorsView = new PointsView(x1,x2);
    _tensorsView.setOrientation(PointsView.Orientation.X1DOWN_X2RIGHT);
    _tensorsView.setLineColor(Color.YELLOW);

    // Plot frame.
    _frame = new PlotFrame(_panel);
    _frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    _frame.setFontSize(fontSize);
    _frame.setSize(width,height);
    _frame.setVisible(true);

    makeModesMenusAndToolBar();
  }

  static float[][] readData() {
    //String fn = "/Users/lh/pythonCode/S043_norm.txt";
    //int n2 = 155, n1 = 175;
    //String fn = "/Users/lh/pythonCode/S042_norm.txt";
    //int n2 = 180, n1 = 90;
    /*String fn = "/Users/lh/pythonCode/S096_norm_743.txt";
    int n2 = 260, n1 = 134;
    float[][] image = zerofloat(n1,n2);
    try {
      Scanner inFile1 = new Scanner(new java.io.File(fn));
      for (int i1=0; i1<n1; ++i1) 
        for (int i2=0; i2<n2; ++i2)  
          image[i2][i1] = inFile1.nextFloat();
    } catch (java.io.IOException e) {
      System.out.println("failed to read");
    }*/
    
    String fn = "/Users/lh/data/images/ultrasound1.png";
    File fnew = new File(fn);
    float[][] image = zerofloat(1, 1);
    try {
      BufferedImage originalImage = ImageIO.read(fnew);
      int n2 = originalImage.getWidth(), n1 = originalImage.getHeight();
      image = zerofloat(n1, n2);
      int rgb, r, g, b;
      for (int i1=0; i1<n1; ++i1) 
        for (int i2=0; i2<n2; ++i2) {
	  rgb = originalImage.getRGB(i2, i1);
	  Color color = new Color(rgb);
	  r = color.getRed();
	  g = color.getGreen();
	  b = color.getBlue();
	  image[i2][i1] = (r + g + b) / 765.0f;
	}
    } catch (java.io.IOException e) {
      System.out.println("failed to read");
    }
     
    return image;
  }
  
  public static List<Mat> cluster(Mat cutout, int k) {
    Mat samples = cutout.reshape(1, cutout.cols() * cutout.rows());
    Mat samples32f = new Mat();
    samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);
    Mat labels = new Mat();
    TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
    Mat centers = new Mat();
    Core.kmeans(samples32f, k, labels, criteria, 1, 
		Core.KMEANS_PP_CENTERS, centers);		
    return showClusters(cutout, labels, centers);
  }

  private static List<Mat> showClusters(Mat cutout, Mat labels, Mat centers) {
     centers.convertTo(centers, CvType.CV_8UC1, 255.0);
     centers.reshape(1);
		
     List<Mat> clusters = new ArrayList<Mat>();
     for(int i = 0; i < centers.rows(); i++) 
       clusters.add(Mat.zeros(cutout.size(), cutout.type()));
     Map<Integer, Integer> counts = new HashMap<Integer, Integer>();
     for(int i = 0; i < centers.rows(); i++) counts.put(i, 0);	
     int rows = 0;
     for(int y = 0; y < cutout.rows(); y++) 
       for(int x = 0; x < cutout.cols(); x++) {
	 int label = (int)labels.get(rows, 0)[0];
	 int r = 255;//(int)centers.get(label, 0)[0];
	 //int g = (int)centers.get(label, 1)[0];
	 //int b = (int)centers.get(label, 0)[0];
	 counts.put(label, counts.get(label) + 1);
	 clusters.get(label).put(y, x, r, r, r);
	 rows++;
       }
     System.out.println(counts);
     return clusters;
  }

  ///////////////////////////////////////////////////////////////////////////
  // private

  private static float SIGMA = 3.0f;

  private int _n1,_n2,_nv;
  private float[][] _image;
  private float _valueMin,_valueMax;
  private StructureTensors _st;

  private PlotPanel _panel;
  private PlotFrame _frame;
  private PixelsView _imageView;
  private PointsView _tensorsView;
  private ColorMap _colorMap;
  private ColorBar _colorBar;

  private void makeModesMenusAndToolBar() {

    // Modes.
    ModeManager mm = _frame.getModeManager();
    TileZoomMode tzm = _frame.getTileZoomMode();

    // Menus.
    JMenu fileMenu = new JMenu("File");
    fileMenu.setMnemonic('F');
    fileMenu.add(new SaveAsPngAction(_frame)).setMnemonic('a');
    fileMenu.add(new ExitAction()).setMnemonic('x');
    JMenu modeMenu = new JMenu("Mode");
    modeMenu.setMnemonic('M');
    modeMenu.add(new ModeMenuItem(tzm));
    JMenu viewMenu = new JMenu("View");
    viewMenu.add(new JCheckBoxMenuItem(new ShowTensorsAction()));
    JMenu structureMenu = new JMenu("Structure");
    JMenuItem isotropicItem = new JRadioButtonMenuItem(
      new AbstractAction("Isotropic") {
        public void actionPerformed(ActionEvent e) {
          updateStructureTensors(0.0f,0.0f,0.0f);
        }
      });
    JMenuItem linearItem = new JRadioButtonMenuItem(
      new AbstractAction("Linear") {
        public void actionPerformed(ActionEvent e) {
          updateStructureTensors(0.0f,100.0f,1.0f);
        }
      });
    JMenuItem layersItem = new JRadioButtonMenuItem(
      new AbstractAction("Layers") {
        public void actionPerformed(ActionEvent e) {
          updateStructureTensors(1.0f,1.0f,1.0f);
        }
      });
    JMenuItem interfacesItem = new JRadioButtonMenuItem(
      new AbstractAction("Interfaces") {
        public void actionPerformed(ActionEvent e) {
          updateStructureTensors(0.0f,1.0f,1.0f);
        }
      });
    structureMenu.add(isotropicItem);
    structureMenu.add(linearItem);
    structureMenu.add(layersItem);
    structureMenu.add(interfacesItem);
    ButtonGroup structureGroup = new ButtonGroup();
    structureGroup.add(isotropicItem);
    structureGroup.add(linearItem);
    structureGroup.add(layersItem);
    structureGroup.add(interfacesItem);
    layersItem.setSelected(true);
    JMenuBar menuBar = new JMenuBar();
    menuBar.add(fileMenu);
    menuBar.add(modeMenu);
    menuBar.add(viewMenu);
    menuBar.add(structureMenu);
    _frame.setJMenuBar(menuBar);

    // Tool bar.
    JToolBar toolBar = new JToolBar(SwingConstants.VERTICAL);
    toolBar.setRollover(true);
    toolBar.add(new ModeToggleButton(tzm));
  }

  public void showTensors(boolean show) {
    if (show) {
      _panel.getTile(0,0).addTiledView(_tensorsView);
    } else {
      _panel.getTile(0,0).removeTiledView(_tensorsView);
    }
  }

  // Actions.
  private class ExitAction extends AbstractAction {
    private ExitAction() {
      super("Exit");
    }
    public void actionPerformed(ActionEvent event) {
      System.exit(0);
    }
  }
  private class SaveAsPngAction extends AbstractAction {
    private PlotFrame _plotFrame;
    private SaveAsPngAction(PlotFrame plotFrame) {
      super("Save as PNG");
      _plotFrame = plotFrame;
    }
    public void actionPerformed(ActionEvent event) {
      JFileChooser fc = new JFileChooser(System.getProperty("user.dir"));
      fc.showSaveDialog(_plotFrame);
      File file = fc.getSelectedFile();
      if (file!=null) {
        String filename = file.getAbsolutePath();
        _plotFrame.paintToPng(300,6,filename);
      }
    }
  }
  private class ShowTensorsAction extends AbstractAction {
    private ShowTensorsAction() {
      super("Tensors");
    }
    public void actionPerformed(ActionEvent event) {
      _show = !_show;
      showTensors(_show);
    }
    boolean _show = false;
  }
  private void updateStructureTensors(float alpha, float beta, float gamma) {
    _st = new StructureTensors(SIGMA,alpha,beta,gamma,_image);
    float[][][] x12 = getTensorEllipses(_n1,_n2,5,_st);
    float[][] x1 = x12[0];
    float[][] x2 = x12[1];
    _tensorsView.set(x1,x2);
  }

  private static float[][][] getTensorEllipses(
    int n1, int n2, int ns, EigenTensors2 et) 
  {
    int nt = 51;
    int m1 = 1+(n1-1)/ns;
    int m2 = 1+(n2-1)/ns;
    int j1 = (n1-1-(m1-1)*ns)/2;
    int j2 = (n2-1-(m2-1)*ns)/2;
    int nm = m1*m2;
    //double r = 0.45*ns;
    float[][] sm = new float[n2][n1];
    for (int i2=0; i2<n2; ++i2) {
      for (int i1=0; i1<n1; ++i1) {
        float[] s = et.getEigenvalues(i1,i2);
        sm[i2][i1] = s[1];
      }
    }
    sm = copy(m1,m2,j1,j2,ns,ns,sm);
    float smq = Quantiler.estimate(0.05f,sm);
    double r = 0.45*ns*sqrt(smq);
    float[][] x1 = new float[nm][nt];
    float[][] x2 = new float[nm][nt];
    double dt = 2.0*PI/(nt-1);
    double ft = 0.0f;
    for (int i2=j2,im=0; i2<n2; i2+=ns) {
      double y2 = i2+r;
      for (int i1=j1; i1<n1; i1+=ns,++im) {
        float[] u = et.getEigenvectorU(i1,i2);
        float[] s = et.getEigenvalues(i1,i2);
        double u1 = u[0];
        double u2 = u[1];
        double v1 = -u2;
        double v2 =  u1;
        double su = s[0];
        double sv = s[1];
        su = max(su,smq);
        sv = max(sv,smq);
        double a = r/sqrt(sv);
        double b = r/sqrt(su);
        for (int it=0; it<nt; ++it) {
          double t = ft+it*dt;
          double cost = cos(t);
          double sint = sin(t);
          x1[im][it] = (float)(i1+b*cost*u1-a*sint*u2);
          x2[im][it] = (float)(i2+a*sint*u1+b*cost*u2);
        }
      }
    }
    return new float[][][]{x1,x2};
  } 

  private static class StructureTensors 
    extends EigenTensors2
  {
    StructureTensors(float sigma, float[][] x) {
      this(sigma,-1.0f,1.0f,1.0f,x);
    }
    StructureTensors(
      float sigma, float alpha, float beta, float gamma, float[][] x) 
    {
      super(x[0].length,x.length);
      int n1 = x[0].length;
      int n2 = x.length;
      float[][] u1 = new float[n2][n1];
      float[][] u2 = new float[n2][n1];
      float[][] su = new float[n2][n1];
      float[][] sv = new float[n2][n1];
      LocalOrientFilter lof = new LocalOrientFilter(sigma);
      lof.apply(x,null,u1,u2,null,null,su,sv,null);
      float[][] sa = pow(su,alpha);
      float[][] sb = pow(div(sv,su),beta);
      float[][] val = coherence(sigma,x);
      //float[][] val = zerofloat(n1,n2);
      float[][] sc = pow(sub(1.0f,val),gamma);
      su = mul(sa,sc);
      sv = mul(sb,su);
      //SimplePlot.asPixels(su);
      //SimplePlot.asPixels(sv);
      for (int i2=0; i2<n2; ++i2) {
        for (int i1=0; i1<n1; ++i1) {
          //setEigenvalues(i1,i2,su[i2][i1],sv[i2][i1]);
          setEigenvalues(i1,i2,0,1.0f);
          setEigenvectorU(i1,i2,u1[i2][i1],u2[i2][i1]);
        }
      }
    }
  }

  private static float[][] coherence(double sigma, float[][] x) {
    int n1 = x[0].length;
    int n2 = x.length;
    LocalOrientFilter lof1 = new LocalOrientFilter(sigma);
    LocalOrientFilter lof2 = new LocalOrientFilter(sigma*4);
    float[][] u11 = new float[n2][n1];
    float[][] u21 = new float[n2][n1];
    float[][] su1 = new float[n2][n1];
    float[][] sv1 = new float[n2][n1];
    float[][] u12 = new float[n2][n1];
    float[][] u22 = new float[n2][n1];
    float[][] su2 = new float[n2][n1];
    float[][] sv2 = new float[n2][n1];
    lof1.apply(x,null,u11,u21,null,null,su1,sv1,null);
    lof2.apply(x,null,u12,u22,null,null,su2,sv2,null);
    float[][] c = u11;
    for (int i2=0; i2<n2; ++i2) {
      for (int i1=0; i1<n1; ++i1) {
        float u11i = u11[i2][i1];
        float u21i = u21[i2][i1];
        float su1i = su1[i2][i1];
        float sv1i = sv1[i2][i1];
        float u12i = u12[i2][i1];
        float u22i = u22[i2][i1];
        float su2i = su2[i2][i1];
        float sv2i = sv2[i2][i1];
        float s111 = (su1i-sv1i)*u11i*u11i+sv1i;
        float s121 = (su1i-sv1i)*u11i*u21i     ;
        float s221 = (su1i-sv1i)*u21i*u21i+sv1i;
        float s112 = (su2i-sv2i)*u12i*u12i+sv2i;
        float s122 = (su2i-sv2i)*u12i*u22i     ;
        float s222 = (su2i-sv2i)*u22i*u22i+sv2i;
        float s113 = s111*s112+s121*s122;
        float s223 = s121*s122+s221*s222;
        float t1 = s111+s221;
        float t2 = s112+s222;
        float t3 = s113+s223;
        float t12 = t1*t2;
        c[i2][i1] = (t12>0.0f)?t3/t12:0.0f;
      }
    }
    return c;
  }

  public static Image toBufferedImage(Mat m){
    int type = BufferedImage.TYPE_BYTE_GRAY;
    if (m.channels() > 1) {
      type = BufferedImage.TYPE_3BYTE_BGR;
    }
    System.out.println("ah!");
    int bufferSize = m.channels()*m.cols()*m.rows();
    byte[] b = new byte[bufferSize];
    m.get(0, 0, b); // get all the pixels
    System.out.println(m.cols());
    System.out.println(m.rows());
    System.out.println(type);
    BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
    final byte[] targetPixels = ((DataBufferByte)image.getRaster().
		    getDataBuffer()).getData();
    System.arraycopy(b, 0, targetPixels, 0, b.length);  
    return image;
  }

  public static void medianFilter(float[][] x, float[][] y, int hw) {
    int n2 = x.length;
    int n1 = x[0].length;
    int w = hw * 2 + 1;
    MedianFinder mf = new MedianFinder(w*w);
    for (int i2=hw; i2<n2-hw; ++i2)
      for (int i1=hw; i1<n1-hw; ++i1) {
        float[] arr = new float[w*w];
	int k = 0;
	for (int i=-hw; i<=hw; ++i)
	  for (int j=-hw; j<=hw; ++j) {
	    arr[k] = x[i2+i][i1+j];
	    k++;
	  } 
	y[i2][i1] = mf.findMedian(arr);
      }
  }

  public static void main(String[] args) {
    System.load("/Users/lh/opencv/build/lib/libopencv_java320.dylib");
    float[][] image = readData();
    TestJtk ip = new TestJtk(image);
    LocalSmoothingFilter lsf = new LocalSmoothingFilter();
    int n2 = image.length;
    int n1 = image[0].length;
    float[][] img = new float[n2][];
    for (int i2=0; i2<n2; ++i2) img[i2] = new float[n1];
    float sigma = 7.0f;
    float alpha = 0.0f, beta = 1.0f, gamma = 1.0f;
    StructureTensors dt = new StructureTensors(
      sigma, alpha, beta, gamma, image); 
    lsf.apply(dt, 5, image, img);
    SimplePlot sp1 = SimplePlot.asPixels(image);
    sp1.setTitle("Original");
    SimplePlot sp2 = SimplePlot.asPixels(img);
    sp2.setTitle("SOS");
    SimplePlot sp3 = SimplePlot.asPixels(sub(img,image));
    sp3.setTitle("SOS - Original");
    // other smoothings
    /*RecursiveGaussianFilter rgf = new RecursiveGaussianFilter(1);
    rgf.apply00(image, img);
    SimplePlot sp3 = SimplePlot.asPixels(img);
    sp3.setTitle("RGF");
    medianFilter(image, img, 1);
    SimplePlot sp4 = SimplePlot.asPixels(img);
    sp4.setTitle("median");*/
    // segment by clustering of img
    /*byte[] tempImg = new byte[n1*n2];
    for (int i2=0; i2<n2; ++i2)
      for (int i1=0; i1<n1; ++i1) 
	tempImg[i2*n1+i1] = (byte)(img[i2][i1]*255);
    Mat mat = new Mat(n2, n1, CvType.CV_8U);
    mat.put(0, 0, tempImg);
    System.out.println(mat.dump());
    Image imageToShow = toBufferedImage(mat);
    EventQueue.invokeLater(new Runnable() {
      public void run(){
	ImageFrame frame = new ImageFrame(imageToShow);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
      }
    })
    Mat clusters = cluster(mat, 3).get(2);
    Image image1 = toBufferedImage(clusters);*/
    /*EventQueue.invokeLater(new Runnable() {
      public void run(){
           ImageFrame frame1 = new ImageFrame(image1);
                frame1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame1.setVisible(true);
            }
    });*/
  }

};
